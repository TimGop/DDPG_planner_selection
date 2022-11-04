import torch
import torch.nn as nn

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO forward() without loops


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
        # f_state is a batch of states
        if type(f_state) is list:
            a_list = []
            t_list = []
            for i in range(len(f_state)):
                x = f_state[i]  # img
                x = x.to(device)
                x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
                # added additional state info below for linear layer (batch)
                x_additional = f_state_additional[i].reshape(1, -1)
                x_Final_Layer = torch.cat((x, x_additional), dim=-1)
                a_list.append(torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
                t_list.append(torch.relu(self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
            return a_list, t_list
            # for i_state in f_state:
            #     x = i_state[0]  # img
            #     x = x.to(device)
            #     x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
            #     # added additional state info below for linear layer (batch)
            #     x_additional = torch.cat((i_state[2], i_state[3], i_state[4]))
            #     x_additional = x_additional.reshape(1, -1)
            #     x_Final_Layer = torch.cat((x, x_additional), dim=-1)
            #     a_list.append(torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
            #     t_list.append(torch.relu(self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
            # return a_list, t_list
        # f_state is a single state
        else:
            x = f_state
            x = x.to(device)
            x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
            # print(f_state_additional)
            # x_additional = torch.cat((f_state_additional[0], f_state_additional[1], f_state_additional[2]))
            x_additional = f_state_additional.reshape(1, -1)  # transpose
            x_Final_Layer = torch.cat((x, x_additional), dim=-1)
            # reminder: state=(img, currentTaskName, maxConsecExecuted, currentlyExecuting, time_left_ep)
            return torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))), torch.relu(
                self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1)))


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

    def forward(self, f_state, action_idx, action_t):
        Q_list = []
        for i_state in f_state:
            x = i_state[0]  # img
            x = x.to(device)
            x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
            # added additional state info below for linear layer (batch)
            x_additional = torch.cat((i_state[2], i_state[3], i_state[4], action_idx, action_t))
            x_additional = x_additional.reshape(1, -1)
            x_Final_Layer = torch.cat((x, x_additional), dim=-1)
            Q_list.append(torch.sigmoid(self.headQ(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
        return Q_list
