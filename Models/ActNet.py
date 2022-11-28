import torch.nn as nn


class ActNet(nn.Module):

    def __init__(self, action_input_size, action_output_size):
        super(ActNet, self).__init__()

        self.act_net = nn.Sequential(nn.Linear(action_input_size, action_input_size * 2),
                                     nn.ReLU(),
                                     nn.Linear(action_input_size * 2, action_output_size),
                                     )

    def forward(self, x):
        act_out = self.act_net(x)

        return act_out
