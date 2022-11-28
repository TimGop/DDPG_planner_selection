import torch
import torch.nn as nn

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, h, w, outputs):
        super(Actor, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(numOutputChannelsConvLayer)
        self.dropout = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()
        # self.in_layer = nn.Linear(1, 100)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer)
        # self.hidden = nn.Linear(100, 20)
        self.headPlanner = nn.Linear(linear_input_size, outputs)

    def forward(self, f_state):
        x = f_state
        x.to(device)
        # x = x.view((-1, 1))
        return torch.softmax(self.headPlanner(self.flatten(self.dropout(self.batchNormalisation(torch.relu(self.conv2d(x)))))), dim=1)


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1, NumAdditionalArgsLinLayer=17):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        # self.in_layer = nn.Linear(18, 100)
        # self.hidden = nn.Linear(100, 20)  # numoutputs --> single value
        self.preHeadQ = nn.Linear(linear_input_size, 100)
        self.headQ = nn.Linear(100, outputs)  # numoutputs --> single value

    def forward(self, f_state, action):
        x = f_state
        x.to(device)
        # x = x.view((128, 1))
        x = self.flatten(self.dropout(self.batchNormalisation(torch.relu(self.conv2d(x)))))
        x_Final_Layer = torch.cat((x, action), dim=1)
        return self.headQ(torch.relu(self.preHeadQ(x_Final_Layer)))
