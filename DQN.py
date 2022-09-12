import torch.nn as nn
import torch
import numpy as random
import random
from collections import deque


class Memory():

    def __init__(self, memsize):
        self.memsize = memsize
        self.memory = deque(maxlen=self.memsize)

    def add_episode(self, epsiode):
        self.memory.append(epsiode)

    def get_batch(self, bsize, time_step):
        sampled_epsiodes = random.sample(self.memory, bsize)
        batch = []
        for episode in sampled_epsiodes:
            point = np.random.randint(0, len(episode) + 1 - time_step)
            batch.append(episode[point:point + time_step])
        return batch


class DQN(nn.Module):

    def __init__(self, input_size):
        super(DQN, self).__init__()

        self.qnetwork = nn.Sequential(nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, 1))

    def forward(self, x):
        qout = self.qnetwork(x)

        return qout
