import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN_net(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN_net, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.dropOut = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten()
        NumAdditionalArgsLinLayer = 35
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.headPlanner = nn.Linear(linear_input_size, outputs)

    def forward(self, state, state_additional):
        state_out = self.flatten(self.dropOut(self.conv2d(state)))
        x_last_layer = torch.concat((state_out, state_additional), dim=1)
        return torch.softmax(self.headPlanner(x_last_layer), dim=1)
