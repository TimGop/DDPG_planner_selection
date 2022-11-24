import torch
import torch.nn as nn

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Actor(nn.Module):
    def __init__(self, h, w, outputs):
        super(Actor, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer)
        self.hidden = nn.Linear(linear_input_size, 20)
        self.headPlanner = nn.Linear(20, outputs)

    def forward(self, f_state):
        x = f_state
        x.to(device)
        x = torch.softmax(
            self.headPlanner(torch.relu(
                self.hidden(self.flatten(self.dropout(self.batchNormalisation(torch.relu(self.conv2d(x)))))))), dim=1)
        return x


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1, NumAdditionalArgsLinLayer=17):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout2d(p=0.5)
        self.flatten = nn.Flatten()
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.hidden = nn.Linear(linear_input_size, 20)  # numoutputs --> single value
        self.headQ = nn.Linear(20, outputs)  # numoutputs --> single value

    def forward(self, f_state, action):
        x = f_state
        x.to(device)
        x = self.flatten(self.dropout(self.batchNormalisation(torch.relu(self.conv2d(x)))))
        x_Final_Layer = torch.cat((x, action), dim=1)
        return self.headQ(torch.relu(self.hidden(x_Final_Layer)))
