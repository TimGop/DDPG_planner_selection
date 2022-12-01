import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=1, out_features=8)
        self.h0 = torch.nn.Linear(in_features=8, out_features=8)
        self.out_layer = torch.nn.Linear(in_features=8, out_features=1)

    def forward(self, f_state):
        x = f_state
        x.to(device)
        x = self.out_layer(torch.relu(self.h0(torch.relu(self.input_layer(x)))))
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.input_layer = torch.nn.Linear(in_features=2, out_features=8)
        self.h0 = torch.nn.Linear(in_features=8, out_features=8)
        self.out_layer = torch.nn.Linear(in_features=8, out_features=1)

    def forward(self, f_state, action):
        x = torch.squeeze((torch.stack((f_state, action), dim=1)))
        x.to(device)
        x = self.out_layer(torch.relu(self.h0(torch.relu(self.input_layer(x)))))
        # x = self.out_layer(self.h0(self.input_layer(x)))
        return x
