import torch
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    # outputs should equal 17
    def __init__(self, h, w, outputs):
        super(Actor, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(128)
        # TODO with kernel size 1 maxpool doesnt do anything???
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = 35
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.headPlanner = nn.Linear(linear_input_size, outputs)  # numoutputs should equal 17 (17 values)
        self.headTime = nn.Linear(linear_input_size, 1)

    def forward(self, f_state, f_state_additional):
        x = f_state
        x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.batchNormalisation(self.conv2d(x)))))
        x_Final_Layer = torch.cat((x, f_state_additional), dim=1)
        # return torch.cat((torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))).max(1)[
        # 1].view( -1, 1),torch.relu( self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1)) ) ), dim=1)
        action = torch.softmax(self.headPlanner(x_Final_Layer), dim=1)
        time = torch.relu(self.headTime(x_Final_Layer)).view(-1)
        return action, time


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.batchNormalisation = nn.BatchNorm2d(128)
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = 35 + 18  # 18 additinal arguments denoting the actions vals and time action
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.headQ = nn.Linear(linear_input_size, outputs)  # numoutputs --> single value
        self.l_ReLu = nn.LeakyReLU(negative_slope=1e-2)

    def forward(self, f_state, f_state_additional, action, time):
        x = f_state
        x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.batchNormalisation(self.conv2d(x)))))
        x_additional = torch.cat((f_state_additional, torch.squeeze(action), time.view(-1, 1)), dim=1)
        x_Final_Layer = torch.cat((x, x_additional), dim=1)
        # TODO does a linear output work with expected rewards bellman equation?
        # below using relu somehow allows time to be reduced by grad. des.
        return self.headQ(x_Final_Layer)
