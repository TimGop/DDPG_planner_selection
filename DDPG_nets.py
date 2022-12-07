import torch
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, h, w, num_planners=17):
        super(Actor, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.50)
        NumAdditionalArgsLinLayer = num_planners * 2 + 2
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer)
        self.h0 = nn.Linear(linear_input_size, 100)
        self.h1 = nn.Linear(100+NumAdditionalArgsLinLayer, 100)
        self.h2 = nn.Linear(100, 100)
        self.headTime = nn.Linear(100, 1)

    def forward(self, f_state, f_state_additional, discrete_action_number):
        x = f_state
        x.to(device)
        x = torch.relu(self.h0(torch.relu(self.flatten(self.dropOut(self.conv2d(x))))))
        x_Final_Layer = torch.cat((x, f_state_additional, discrete_action_number), dim=1)
        time = self.headTime(torch.relu(self.h2(torch.relu(self.h1(x_Final_Layer))))).view(-1)
        return time


class Critic(nn.Module):
    def __init__(self, h, w, outputs=1, num_planners=17):
        super(Critic, self).__init__()
        numOutputChannelsConvLayer = 32
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.50)
        NumAdditionalArgsLinLayer = num_planners * 2 + 3  # action_no, time left and action time
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer)
        self.h0 = nn.Linear(linear_input_size, 100)
        self.h1 = nn.Linear(100+NumAdditionalArgsLinLayer, 100)
        self.h2 = nn.Linear(100, 100)
        self.headQ = nn.Linear(100, outputs)  # numoutputs --> single value

    def forward(self, f_state, f_state_additional, discrete_action_number, action_time):
        x = f_state
        x.to(device)
        x = torch.relu(self.h0(torch.relu(self.dropOut(self.flatten(self.conv2d(x))))))
        x_additional = torch.cat((f_state_additional, discrete_action_number, action_time.view(-1, 1)),
                                 dim=1)
        x_Final_Layer = torch.cat((x, x_additional), dim=1)
        return self.headQ(torch.relu(self.h2(torch.relu(self.h1(x_Final_Layer)))))
