import torch.nn as nn
import torch
import numpy as random
import random
from collections import deque


class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        self.qnetwork = nn.Sequential(nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, input_size),
                                      nn.ReLU(),
                                      nn.Linear(input_size, output_size))

    def forward(self, x):
        qout = self.qnetwork(x)

        return qout
