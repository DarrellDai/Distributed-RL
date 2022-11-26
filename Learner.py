import _pickle as cPickle
import argparse
import os
import time

import redis
import torch
import yaml
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Behavior_Cloning import Behavior_Cloning
from Experience_Replay import Distributed_Memory
from utils import initialize_model, save_checkpoint, load_checkpoint, wait_until_present, \
    calculate_loss_from_all_loss_stats


class Learner:
    def __init__(self, memsize, num_actor=1, epsilon=1, hostname="localhost", device_idx=[0], instance_idx=0):
        self.num_actor = num_actor
        self.memory_size = memsize
        self.device_idx = device_idx
        self.epsilon = epsilon
        self.instance_idx = instance_idx
        if MPI.COMM_WORLD.Get_rank() == 0:
            self._connect = redis.Redis(host=hostname, db=self.instance_idx)
            self._connect.delete("id_to_name")
            self._connect.delete("params")
            self._connect.delete("optimizer")
            self._connect.delete("epsilon")
            self._connect.delete("success_count")
            self._connect.delete("episode_count")
            self._connect.delete("epoch")
            self._connect.delete("experience")
            self._connect.delete("Update")
            self._connect.delete("to_update")
            self._connect.delete("Update params")
            self._connect.delete("Update Experience")

        if device_idx == -1:
            self.device = torch.device('cpu')
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, please choose CPU by setting device_idx=-1")
            self.device = torch.device(
                'cuda:' + str(device_idx[MPI.COMM_WORLD.Get_rank()]))
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

    def initialize_model(self, cnn_out_size, lstm_hidden_size, action_shape, atten_size, method):
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
        self.models = initialize_model(self.agent_ids, cnn_out_size, lstm_hidden_size, action_shape,
                                       atten_size, self.device, method)

    def get_model_state_dicts(self):
        model_state_dicts = {}
        for id in self.agent_ids:
            model_state_dicts[id] = self.models[id].get_model_state_dict()
        return model_state_dicts

    def get_optimizer_state_dicts(self):
        optimizer_state_dicts = {}
        for id in self.agent_ids:
            optimizer_state_dicts[id] = self.models[id].get_optimizer_state_dict()
        return optimizer_state_dicts

    def initialize_training(self, initial_learning_rate, learning_rate_gamma, learning_rate_step_size,
                            checkpoint_to_load=None, resume=False):
        for id in self.agent_ids:
            self.models[id].initialize_training(initial_learning_rate, learning_rate_step_size, learning_rate_gamma)
        if resume:
            self.initial_epoch_count = None
            if MPI.COMM_WORLD.Get_rank() == 0:
                model_state_dicts, optimizer_state_dicts, episode_count, self.epsilon, self.initial_epoch_count, success_count = load_checkpoint(
                    checkpoint_to_load + ".pth.tar", self.device)
                # print("episode_count")
                self._connect.set("episode_count", cPickle.dumps(episode_count))
                # print("Sending epsilon")
                self._connect.set("epsilon", cPickle.dumps(self.epsilon))
                self._connect.set("success_count", cPickle.dumps(success_count))
                self._connect.set("epoch", cPickle.dumps(self.initial_epoch_count))
                self._connect.set("params", cPickle.dumps(model_state_dicts))
                self._connect.set("optimizer", cPickle.dumps(optimizer_state_dicts))

            self.epsilon = MPI.COMM_WORLD.bcast(self.epsilon)
            self.initial_epoch_count = MPI.COMM_WORLD.bcast(self.initial_epoch_count)
            for id in self.agent_ids:
                self.models[id].load_model_state_dict(MPI.COMM_WORLD.bcast(model_state_dicts[id]))
                self.models[id].load_optimizer_state_dict(MPI.COMM_WORLD.bcast(optimizer_state_dicts[id]))

        else:
            if MPI.COMM_WORLD.Get_rank() == 0:
                self._connect.set("episode_count", cPickle.dumps(0))
                self._connect.set("epsilon", cPickle.dumps(1))
                self._connect.set("success_count", cPickle.dumps(0))
                self._connect.set("epoch", cPickle.dumps(0))
                self._connect.set("params", cPickle.dumps(self.get_model_state_dicts()))
            self.initial_epoch_count = 0

    def train(self, batch_size, sequence_length, final_epsilon, epsilon_vanish_rate, initial_learning_rate,
              learning_rate_gamma, learning_rate_step_size, name_tensorboard,
              total_epochs, num_batch_per_learner, actor_update_freq,
              performance_display_interval, checkpoint_save_interval, checkpoint_to_save):
        if MPI.COMM_WORLD.Get_rank() == 0:
            writer = SummaryWriter(os.path.join("runs", str(self.instance_idx) + "_" + name_tensorboard +
                                                "_" + str(batch_size) + "_" + str(num_batch_per_learner) + "_" + str(
                initial_learning_rate) + "_" + str(learning_rate_gamma) + "_" + str(learning_rate_step_size)))
            self._wait_memory()
            counter = tqdm(range(self.initial_epoch_count, total_epochs))
        else:
            counter = range(self.initial_epoch_count, total_epochs)
        loss_stats = {}
        batches = None

        for epoch in counter:
            for id in self.agent_ids:
                loss_stats[id] = {}
            if MPI.COMM_WORLD.Get_rank() == 0:
                batches = self._memory.get_batch(bsize=batch_size, num_learner=MPI.COMM_WORLD.Get_size(),
                                                 num_batch=num_batch_per_learner,
                                                 sequence_length=sequence_length)
            batches_per_learner = MPI.COMM_WORLD.scatter(batches)
            for id in self.agent_ids:
                loss_stat = self.models[id].learn(batches_per_learner[id], epoch)
                for key in loss_stat:
                    loss_stats[id][key] = loss_stat[key]

            loss_stats_all_learner = MPI.COMM_WORLD.gather(loss_stats)
            if MPI.COMM_WORLD.Get_rank() == 0:
                loss = calculate_loss_from_all_loss_stats(loss_stats_all_learner, self.agent_ids)
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
                    optimizer_state_dicts = self.get_optimizer_state_dicts()
                    for id in self.agent_ids:
                        for key in optimizer_state_dicts[id]:
                            print('\n Epoch: [%d | %d] LR (%s): %.16f Epsilon : %f \n' % (
                                epoch, total_epochs, key, optimizer_state_dicts[id][key]['param_groups'][0]["lr"],
                                self.epsilon))
                        for key in loss[id]:
                            print('\n Agent %d, Loss (%s): %f \n' % (id, key, loss[id][key]))
                for id in self.agent_ids:
                    # todo: success_count should be associated to player type
                    if episode_count != 0:
                        writer.add_scalar(self.id_to_name[id] + ": Success Rate vs Epoch",
                                          success_count / episode_count,
                                          epoch)
                    for loss_name in loss[id]:
                        writer.add_scalar(self.id_to_name[id] + ": Loss vs Epoch", loss[id][loss_name], epoch)
                writer.flush()

                if epoch % actor_update_freq == 0:
                    with self._connect.lock("Update params"):
                        self._connect.set("params", cPickle.dumps(self.get_model_state_dicts()))
                        self._connect.set("to_update", cPickle.dumps(True))
                if self.epsilon > final_epsilon:
                    self.epsilon *= epsilon_vanish_rate

                # #todo:not only first learner do this, but all
                # if (epoch + 1) % target_update_freq == 0 and self.method == "DQN":
                #     for id in self.agent_ids:
                #         self.target_model[id].load_state_dict(self.models[id].state_dict())
                if (epoch + 1) % checkpoint_save_interval == 0:
                    model_state_dicts = self.get_model_state_dicts()
                    optimizer_state_dicts = self.get_optimizer_state_dicts()
                    save_checkpoint({
                        'model_state_dicts': model_state_dicts,
                        'optimizer_state_dicts': optimizer_state_dicts,
                        'epsilon': self.epsilon,
                        "episode_count": episode_count,
                        "epoch_count": epoch,
                        "success_count": success_count
                    }, filename=str(self.instance_idx) + "_" + str(epoch + 1) + "_" + checkpoint_to_save +
                                "_" + str(batch_size) + "_" + str(num_batch_per_learner) + "_" + str(
                        initial_learning_rate) + "_" + str(learning_rate_gamma) + "_" + str(
                        learning_rate_step_size) + ".pth.tar")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learner process for distributed reinforcement.')
    parser.add_argument('-r', '--redisserver', type=str, default='localhost', help="Redis's server name.")
    parser.add_argument('-ins', '--instance_idx', type=int, default=0, help="The index of instance to run")
    parser.add_argument('-mc', '--model_config', type=str, default='Model.yaml', help="Model config file name")
    parser.add_argument('-rc', '--run_config', type=str, default='Train.yaml', help="Running config file name")
    args = parser.parse_args()
    with open("Config/" + args.model_config) as file:
        model_param = yaml.safe_load(file)
    with open("Config/" + args.run_config) as file:
        run_param = yaml.safe_load(file)
    learner = Learner(memsize=run_param["memory_size"], epsilon=run_param["initial_epsilon"],
                      hostname=args.redisserver, device_idx=run_param["device_idx"], instance_idx=args.instance_idx)
    learner.initialize_model(cnn_out_size=model_param["cnn_out_size"], lstm_hidden_size=model_param["lstm_hidden_size"],
                             action_shape=model_param["action_shape"],
                             atten_size=model_param["atten_size"], method=Behavior_Cloning)
    learner.initialize_training(initial_learning_rate=run_param["initial_learning_rate"],
                                learning_rate_gamma=run_param["learning_rate_gamma"],
                                learning_rate_step_size=run_param["learning_rate_step_size"],
                                resume=run_param["resume"],
                                checkpoint_to_load=run_param["checkpoint_to_load"])
    learner.train(batch_size=run_param["batch_size"], sequence_length=run_param["sequence_length"],
                  name_tensorboard=run_param["name_tensorboard"],
                  final_epsilon=run_param["final_epsilon"],
                  epsilon_vanish_rate=run_param["epsilon_vanish_rate"],
                  initial_learning_rate=run_param["initial_learning_rate"],
                  learning_rate_gamma=run_param["learning_rate_gamma"],
                  learning_rate_step_size=run_param["learning_rate_step_size"],
                  total_epochs=run_param["total_epochs"], num_batch_per_learner=run_param["num_batch_per_learner"],
                  actor_update_freq=run_param["actor_update_freq(epochs)"],
                  performance_display_interval=run_param["performance_display_interval(epochs)"],
                  checkpoint_save_interval=run_param["checkpoint_save_interval(epochs)"],
                  checkpoint_to_save=run_param["checkpoint_to_save"])
