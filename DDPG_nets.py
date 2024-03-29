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
        x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
        x_Final_Layer = torch.cat((x, f_state_additional), dim=1)
        return torch.cat((torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))).max(1)[1].view(
            -1, 1), torch.relu(
            self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1)))), dim=1)


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = 35 + 2  # 2 additinal arguments denoting the action taken
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.headQ = nn.Linear(linear_input_size, outputs)  # numoutputs --> single value

    def forward(self, f_state, f_state_additional, action):
        x = f_state
        x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
        x_additional = torch.cat((f_state_additional, torch.squeeze(action)), dim=1)
        x_Final_Layer = torch.cat((x, x_additional), dim=1)
        return torch.sigmoid(self.headQ(x_Final_Layer.view(x_Final_Layer.size(0), -1)))
