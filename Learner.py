import numpy as np
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

import redis
import os
import time
from copy import deepcopy
from tqdm import tqdm
import _pickle as cPickle
import yaml
import argparse

from utils import initialize_model, wrap_model_with_dataparallel, save_checkpoint, load_checkpoint
from Experience_Replay import Distributed_Memory


class Learner:
    def __init__(self, id_to_name, memsize, hostname="localhost", device_idx=[0]):
        self.id_to_name = id_to_name
        self.agent_ids = tuple(id_to_name.keys())
        self.device_idx = device_idx
        self._connect = redis.Redis(host=hostname)
        self._connect.delete("params")
        self._connect.delete("epsilon")
        self._connect.delete("received")
        self._connect.delete("Pull params")
        self._connect.delete("Update")
        self._connect.delete("episode_count")
        self._memory = Distributed_Memory(memsize, self.agent_ids, connect=redis.Redis(host=hostname))
        self._memory.start()
        self.device = torch.device('cuda:' + str(device_idx[0]) if torch.cuda.is_available() else 'cpu')

    def _wait_memory(self):
        # print("Waiting for memory")
        while True:
            if len(self._memory) == self._memory.memsize:
                # print("Memory got!")
                break
            time.sleep(0.1)

    def _sleep(self):
        mlen = self._connect.llen("experience")
        time.sleep(0.01 * mlen)

    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, action_out_size, atten_size):
        self.main_model = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape,
                                           action_out_size,
                                           atten_size)

        wrap_model_with_dataparallel(self.main_model, self.device_idx)
        self.target_model = deepcopy(self.main_model)

    def get_model_state_dict(self):
        model_state_dict = {}
        for id in self.agent_ids:
            model_state_dict[id] = self.main_model[id].state_dict()
        return model_state_dict

    def initialize_training(self, learning_rate, checkpoint_to_load=None, resume=False):
        self.criterion = nn.MSELoss()
        self.optimizer = {}
        for id in self.agent_ids:
            self.optimizer[id] = torch.optim.Adam(self.main_model[id].parameters(), lr=learning_rate)
        if resume:
            model_state_dicts, optimizer_state_dicts, episode_count, epsilon, self.initial_epoch_count, success_count = load_checkpoint(
                checkpoint_to_load, self.device)
            # print("episode_count")
            self._connect.set("episode_count", cPickle.dumps(episode_count))
            # print("Sending epsilon")
            self._connect.set("epsilon", cPickle.dumps(epsilon))
            self._connect.set("success_count", cPickle.dumps(success_count))
            for id in self.agent_ids:
                self.main_model[id].load_state_dict(model_state_dicts[id])
                self.optimizer[id].load_state_dict(optimizer_state_dicts[id])
            self.target_model = deepcopy(self.main_model)
        else:
            self.initial_epoch_count=0
            # print("episode_count")
            self._connect.set("episode_count", cPickle.dumps(0))
            # print("Sending epsilon")
            self._connect.set("epsilon", cPickle.dumps(1))
            self._connect.set("success_count", cPickle.dumps(0))
        self._connect.set("params", cPickle.dumps(self.get_model_state_dict()))

    def train(self, batch_size, time_step, gamma, learning_rate, name_tensorboard, total_epochs, target_update_freq,
              checkpoint_save_interval, checkpoint_to_save):

        writer = SummaryWriter(os.path.join("runs", name_tensorboard))
        loss_stat = {}
        self._wait_memory()
        for epoch in tqdm(range(self.initial_epoch_count, total_epochs)):
            for id in self.agent_ids:
                loss_stat[id] = []
            self.learn(batch_size, time_step, gamma, loss_stat)
            self._connect.set("params", cPickle.dumps(self.get_model_state_dict()))
            print('\n Epoch: [%d | %d] LR: %f \n' % (epoch, total_epochs, learning_rate))
            loss = {}
            for id in self.agent_ids:
                if len(loss_stat[id]) > 0:
                    loss[id] = np.mean(loss_stat[id])
                else:
                    loss[id] = 0
                print('\n Agent %d, Loss: %f \n' % (id, loss[id]))
                writer.add_scalar(self.id_to_name[id] + ": Loss/train", loss[id], epoch)
            writer.flush()
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
                    'epsilon': cPickle.loads(self._connect.get("epsilon")),
                    "episode_count": cPickle.loads(self._connect.get("episode_count")),
                    "epoch_count": epoch,
                    "success_count": cPickle.loads(self._connect.get("success_count"))
                }, filename=checkpoint_to_save)

            self._sleep()

    def learn(self, batch_size, time_step, gamma, loss_stat):
        for id in self.agent_ids:
            batches = self._memory.get_batch(bsize=batch_size, time_step=time_step, agent_id=id)
            for batch in batches:
                hidden_batch, cell_batch, out_batch = self.main_model[id].module.lstm.init_hidden_states_and_outputs(
                    bsize=len(batch))
                hidden_batch.to(self.device)
                cell_batch.to(self.device)
                out_batch.to(self.device)
                current_visual_obs = []
                current_vector_obs = []
                act = []
                rewards = []
                next_visual_obs = []
                next_vector_obs = []

                for b in batch:
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
                # update params
                self.optimizer[id].step()


if __name__ == "__main__":
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-c', '--config', type=str, default='Train.yaml', help="Config file name")
    args = parser.parse_args()
    with open("Config/"+args.config) as file:
        param = yaml.safe_load(file)
    learner = Learner(id_to_name=param["id_to_name"], memsize=param["memory_size"], hostname=args.redisserver, device_idx=param["device_idx"])
    learner.initialize_model(cnn_out_size=param["cnn_out_size"], lstm_hidden_size=param["lstm_hidden_size"],
                             action_shape=param["action_shape"],
                             action_out_size=param["action_out_size"], atten_size=param["atten_size"])
    learner.initialize_training(learning_rate=param["learning_rate"], resume=param["resume"],
                                checkpoint_to_load=param["checkpoint_to_load"])
    learner.train(batch_size=param["batch_size"], time_step=param["time_step"], gamma=param["gamma"],
                  learning_rate=param["learning_rate"], name_tensorboard=param["name_tensorboard"],
                   total_epochs=param["total_epochs"], target_update_freq=param["target_update_freq(epochs)"],
                  checkpoint_save_interval=param["checkpoint_save_interval"],
                  checkpoint_to_save=param["checkpoint_to_save"])
