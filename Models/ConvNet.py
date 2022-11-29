import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self,cnn_out_size):
        super().__init__()
        # convolutional neural networks
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8,
                                     stride=4)  # potential check - in_channels
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=7, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv_out = self.conv_layer1(x)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer2(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer3(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.conv_layer4(conv_out)
        conv_out = self.relu(conv_out)
        return conv_out