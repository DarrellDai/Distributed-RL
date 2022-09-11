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

    def __init__(self, input_size, out_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.out_size = out_size

        self.adv = nn.Linear(in_features=input_size, out_features=self.out_size)
        self.val = nn.Linear(in_features=input_size, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x, bsize):
        adv_out = self.adv(x)
        val_out = self.val(x)

        qout = val_out.expand(bsize, self.out_size) + (
                adv_out - adv_out.mean(dim=1).unsqueeze(dim=1).expand(bsize, self.out_size))

        return qout
