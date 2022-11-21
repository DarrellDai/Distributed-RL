import torch.nn as nn
import numpy as np

class BCNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.bc_net = nn.Sequential(nn.Linear(input_size, int(input_size / 4)),
                                    nn.ReLU(),
                                    nn.Linear(int(input_size / 4), output_size),
                                    nn.Softmax(-1)
                                    )

    def forward(self, x):
        x = self.bc_net(x)
        return x
