import torch
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, h, w, num_planners=17):
        super(Actor, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(numOutputChannelsConvLayer)
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = num_planners * 2 + 1
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.preHeadPlanner = nn.Linear(linear_input_size, 100)
        self.headPlanner = nn.Linear(100, num_planners)  # numoutputs should equal 17 (17 values)
        self.preHeadTime = nn.Linear(linear_input_size, 100)
        self.headTime = nn.Linear(100, num_planners)

    def forward(self, f_state, f_state_additional):
        x = f_state
        x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.batchNormalisation(torch.relu(self.conv2d(x))))))
        x_Final_Layer = torch.cat((x, f_state_additional), dim=1)
        action = torch.softmax(self.headPlanner(torch.relu(self.preHeadPlanner(x_Final_Layer))), dim=1)
        time = self.headTime(torch.relu(self.preHeadTime(x_Final_Layer)))#.view(-1)
        return action, time


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1, num_planners=17):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(32)
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = num_planners * 2 + (2 * num_planners + 1)  # time left
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.preHeadQ = nn.Linear(linear_input_size, 100)
        self.headQ = nn.Linear(100, outputs)  # numoutputs --> single value

    def forward(self, f_state, f_state_additional, action, time):
        x = f_state
        x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.batchNormalisation(self.conv2d(x)))))
        x_additional = torch.cat((f_state_additional, torch.squeeze(action), torch.squeeze(time)), dim=1)
        x_Final_Layer = torch.cat((x, x_additional), dim=1)
        return self.headQ(torch.relu(self.preHeadQ(x_Final_Layer)))
