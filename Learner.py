import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import redis
from mpi4py import MPI
import os
import time
from copy import deepcopy
from tqdm import tqdm
import _pickle as cPickle
import yaml
import argparse

from utils import initialize_model, save_checkpoint, load_checkpoint, wait_until_present, sync_grads, \
    calculate_loss_from_all_loss_stats
from Experience_Replay import Distributed_Memory


class Learner:
    def __init__(self, memsize, num_actor=1, epsilon=1, hostname="localhost", device_idx=[0]):
        self.num_actor = num_actor
        self.memory_size = memsize
        self.device_idx = device_idx
        self.epsilon = epsilon
        if MPI.COMM_WORLD.Get_rank() == 0:
            self._connect = redis.Redis(host=hostname)
            self._connect.delete("id_to_name")
            self._connect.delete("params")
            self._connect.delete("epsilon")
            self._connect.delete("success_count")
            self._connect.delete("episode_count")
            self._connect.delete("epoch")
            self._connect.delete("experience")
            self._connect.delete("Update")
            self._connect.delete("Update params")
            self._connect.delete("Update Experience")

        self.device = torch.device(
            'cuda:' + str(device_idx[MPI.COMM_WORLD.Get_rank()]) if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(10)

    def _wait_memory(self):
        last_length = -1
        while True:
            if len(self._memory) != last_length:
                print("Waiting for memory: {}/{}".format(len(self._memory), self._memory.memsize))
            if len(self._memory) == self._memory.memsize:
                # print("Memory got!")
                break
            last_length = len(self._memory)
            time.sleep(0.1)

    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size):
        self.id_to_name = None
        self.agent_ids = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            wait_until_present(self._connect, "id_to_name")
            print("Got id_to_name")
            self.id_to_name = cPickle.loads(self._connect.get("id_to_name"))
            self.agent_ids = tuple(self.id_to_name.keys())
            self._memory = Distributed_Memory(self.memory_size, self.agent_ids, connect=self._connect)
            self._memory.start()
        self.id_to_name = MPI.COMM_WORLD.bcast(self.id_to_name)
        self.agent_ids = MPI.COMM_WORLD.bcast(self.agent_ids)
        self.main_model = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape,
                                           action_out_size,
                                           atten_size)

    def get_model_state_dict(self):
        model_state_dict = {}
        for id in self.agent_ids:
            model_state_dict[id] = self.main_model[id].state_dict()
        return model_state_dict

    def initialize_training(self, learning_rate, checkpoint_to_load=None, resume=False):
        self.criterion = nn.MSELoss()
        self.optimizer = {}
        self.initial_epoch_count = None
        for id in self.agent_ids:
            self.optimizer[id] = torch.optim.Adam(self.main_model[id].parameters(), lr=learning_rate)
        if resume:
            if MPI.COMM_WORLD.Get_rank() == 0:
                model_state_dicts, optimizer_state_dicts, episode_count, self.epsilon, self.initial_epoch_count, success_count = load_checkpoint(
                    checkpoint_to_load, self.device)
                # print("episode_count")
                self._connect.set("episode_count", cPickle.dumps(episode_count))
                # print("Sending epsilon")
                self._connect.set("epsilon", cPickle.dumps(self.epsilon))
                self._connect.set("success_count", cPickle.dumps(success_count))
                self._connect.set("epoch", cPickle.dumps(self.initial_epoch_count))
                for id in self.agent_ids:
                    self.main_model[id].load_state_dict(model_state_dicts[id])
                    self.optimizer[id].load_state_dict(optimizer_state_dicts[id])
                self._connect.set("params", cPickle.dumps(self.get_model_state_dict()))

            self.epsilon = MPI.COMM_WORLD.bcast(self.epsilon)
            self.initial_epoch_count = MPI.COMM_WORLD.bcast(self.initial_epoch_count)
            for id in self.agent_ids:
                self.main_model[id].load_state_dict(MPI.COMM_WORLD.bcast(self.main_model[id].state_dict()))

                self.optimizer[id].load_state_dict(MPI.COMM_WORLD.bcast(self.optimizer[id].state_dict()))
        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                self._connect.set("episode_count", cPickle.dumps(0))
                self._connect.set("epsilon", cPickle.dumps(1))
                self._connect.set("success_count", cPickle.dumps(0))
                self._connect.set("epoch", cPickle.dumps(0))
                self._connect.set("params", cPickle.dumps(self.get_model_state_dict()))
            self.initial_epoch_count = 0
        self.target_model = deepcopy(self.main_model)

    def train(self, batch_size, time_step, gamma, learning_rate, final_epsilon, epsilon_vanish_rate, name_tensorboard,
              total_epochs, actor_update_freq, target_update_freq,
              performance_display_interval, checkpoint_save_interval, checkpoint_to_save):
        if MPI.COMM_WORLD.Get_rank() == 0:
            writer = SummaryWriter(os.path.join("runs", name_tensorboard))
            self._wait_memory()
            counter = tqdm(range(self.initial_epoch_count, total_epochs))
        else:
            counter = range(self.initial_epoch_count, total_epochs)
        loss_stat = {}
        batches = None
        for epoch in counter:
            for id in self.agent_ids:
                loss_stat[id] = []
            if MPI.COMM_WORLD.Get_rank() == 0:
                batches = self._memory.get_batch(bsize=batch_size, num_batch=MPI.COMM_WORLD.Get_size(),
                                                 time_step=time_step)
            batch = MPI.COMM_WORLD.scatter(batches)
            self.learn(batch, time_step, gamma, loss_stat)

            loss_stats = MPI.COMM_WORLD.gather(loss_stat)
            if MPI.COMM_WORLD.Get_rank() == 0:
                loss = calculate_loss_from_all_loss_stats(loss_stats, self.agent_ids)
                with self._connect.lock("Update"):
                    self._connect.set("epoch", cPickle.dumps(epoch))
                    self._connect.set("epsilon", cPickle.dumps(self.epsilon))
                    wait_until_present(self._connect, "success_count")
                    success_count = cPickle.loads(self._connect.get("success_count"))
                    # print("Learner got success_count")
                    wait_until_present(self._connect, "episode_count")
                    episode_count = cPickle.loads(self._connect.get("episode_count"))
                    # print("Learner got episode_count")
                if (epoch + 1) % performance_display_interval == 0:
                    print('\n Epoch: [%d | %d] LR: %f Epsilon : %f \n' % (
                    epoch, total_epochs, learning_rate, self.epsilon))
                    for id in self.agent_ids:
                        print('\n Agent %d, Loss: %f \n' % (id, loss[id]))
                for id in self.agent_ids:
                    # todo: success_count should be associated to player type
                    if episode_count != 0:
                        writer.add_scalar(self.id_to_name[id] + ": Success Rate vs Epoch",
                                          success_count / episode_count,
                                          epoch)
                    writer.add_scalar(self.id_to_name[id] + ": Loss vs Epoch", loss[id], epoch)
                writer.flush()

                if epoch % actor_update_freq == 0:
                    with self._connect.lock("Update params"):
                        self._connect.set("params", cPickle.dumps(self.get_model_state_dict()))
                if self.epsilon > final_epsilon:
                    self.epsilon *= epsilon_vanish_rate

                if (epoch + 1) % target_update_freq == 0:
                    for id in self.agent_ids:
                        self.target_model[id].load_state_dict(self.main_model[id].state_dict())
                if (epoch + 1) % checkpoint_save_interval == 0:
                    model_state_dicts = {}
                    optimizer_state_dicts = {}
                    for id in self.agent_ids:
                        model_state_dicts[id] = self.main_model[id].state_dict()
                        optimizer_state_dicts[id] = self.optimizer[id].state_dict()
                    save_checkpoint({
                        'model_state_dicts': model_state_dicts,
                        'optimizer_state_dicts': optimizer_state_dicts,
                        'epsilon': self.epsilon,
                        "episode_count": episode_count,
                        "epoch_count": epoch,
                        "success_count": success_count
                    }, filename=checkpoint_to_save)

    def learn(self, batch, time_step, gamma, loss_stat):
        for id in self.agent_ids:
            hidden_batch, cell_batch, out_batch = self.main_model[id].lstm.init_hidden_states_and_outputs(
                bsize=len(batch[id]))
            hidden_batch.to(self.device)
            cell_batch.to(self.device)
            out_batch.to(self.device)
            current_visual_obs = []
            current_vector_obs = []
            act = []
            rewards = []
            next_visual_obs = []
            next_vector_obs = []

            for b in batch[id]:
                cvis, cves, ac, rw, nvis, nves = [], [], [], [], [], []
                for element in b:
                    cvis.append(element[0][0])
                    cves.append(element[0][1][list(element[0][1].keys())[0]])
                    ac.append(element[1])
                    rw.append(element[2])
                    nvis.append(element[3][0])
                    nves.append(element[3][1][list(element[0][1].keys())[0]])
                current_visual_obs.append(cvis)
                current_vector_obs.append(cves)
                act.append(ac)
                rewards.append(rw)
                next_visual_obs.append(nvis)
                next_vector_obs.append(nves)

            current_visual_obs = np.array(current_visual_obs)
            current_vector_obs = np.array(current_vector_obs)
            act = np.array(act)
            rewards = np.array(rewards)
            next_visual_obs = np.array(next_visual_obs)
            next_vector_obs = np.array(next_vector_obs)

            current_visual_obs = torch.from_numpy(current_visual_obs).float().to(self.device)
            act = torch.from_numpy(act).long().to(self.device)
            rewards = torch.from_numpy(rewards).float().to(self.device)
            next_visual_obs = torch.from_numpy(next_visual_obs).float().to(self.device)
            visual_obs = torch.concat((current_visual_obs, next_visual_obs[:, -1:]), 1)
            Q_next_max = torch.zeros(len(batch)).float().to(self.device)
            for batch_idx in range(len(batch)):
                _, _, Q_next, _, _ = self.target_model[id](visual_obs[batch_idx:batch_idx + 1],
                                                           act[batch_idx:batch_idx + 1],
                                                           hidden_state=hidden_batch[
                                                                        batch_idx:batch_idx + 1],
                                                           cell_state=cell_batch[
                                                                      batch_idx:batch_idx + 1],
                                                           lstm_out=out_batch[
                                                                    batch_idx:batch_idx + 1])
                Q_next_max[batch_idx] = torch.max(Q_next.reshape(-1))
            target_values = rewards[:, time_step - 1] + (gamma * Q_next_max)
            target_values = target_values.float()

            _, _, Q_s, _, _ = self.main_model[id](current_visual_obs, act,
                                                  hidden_state=hidden_batch, cell_state=cell_batch,
                                                  lstm_out=out_batch)

            Q_s_a = Q_s[:, 0, 0]
            loss = self.criterion(Q_s_a, target_values)

            #  save performance measure
            loss_stat[id].append(loss.item())

            # make previous grad zero
            self.optimizer[id].zero_grad()

            # backward
            loss.backward()

            sync_grads(self.main_model[id])
            # update params
            self.optimizer[id].step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-mc', '--model_config', type=str, default='Model.yaml', help="Model config file name")
    parser.add_argument('-rc', '--run_config', type=str, default='Train.yaml', help="Running config file name")
    args = parser.parse_args()
    with open("Config/" + args.model_config) as file:
        model_param = yaml.safe_load(file)
    with open("Config/" + args.run_config) as file:
        run_param = yaml.safe_load(file)
    learner = Learner(memsize=run_param["memory_size"], epsilon=run_param["initial_epsilon"],
                      hostname=args.redisserver, device_idx=run_param["device_idx"])
    learner.initialize_model(cnn_out_size=model_param["cnn_out_size"], lstm_hidden_size=model_param["lstm_hidden_size"],
                             action_shape=model_param["action_shape"],
                             action_out_size=model_param["action_out_size"], atten_size=model_param["atten_size"])
    learner.initialize_training(learning_rate=run_param["learning_rate"], resume=run_param["resume"],
                                checkpoint_to_load=run_param["checkpoint_to_load"])
    learner.train(batch_size=run_param["batch_size"], time_step=run_param["time_step"], gamma=run_param["gamma"],
                  learning_rate=run_param["learning_rate"], name_tensorboard=run_param["name_tensorboard"],
                  final_epsilon=run_param["final_epsilon"],
                  epsilon_vanish_rate=run_param["epsilon_vanish_rate"],
                  total_epochs=run_param["total_epochs"], actor_update_freq=run_param["actor_update_freq(epochs)"],
                  target_update_freq=run_param["target_update_freq(epochs)"],
                  performance_display_interval=run_param["performance_display_interval(epochs)"],
                  checkpoint_save_interval=run_param["checkpoint_save_interval(epochs)"],
                  checkpoint_to_save=run_param["checkpoint_to_save"])
