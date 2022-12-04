import math

from DQN_net import DQN_net
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(object):
    def __init__(self, batchsize, w, h, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=200, n_actions=17):
        self.BATCH_SIZE = batchsize
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay

        self.n_actions = n_actions

        self.policy_net = DQN_net(w, h, self.n_actions).to(device)
        self.target_net = DQN_net(w, h, self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.policy_net.parameters())

    def set_eval(self):  # sets networks to evaluation mode (faster)
        # Sets the model in evaluation mode
        self.policy_net.eval()
        self.target_net.eval()

    def set_train(self):  # sets networks to training mode (needed for learning but slower)
        # Sets the model in training mode
        self.policy_net.train()
        self.target_net.train()

    def optimize_model(self, transition_batch):
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        state_additional_batch = torch.stack(transition_batch.state_additional).to(device)
        action_batch = torch.stack(transition_batch.action).to(device)
        time_batch = torch.stack(transition_batch.time).to(device)
        reward_batch = torch.cat(transition_batch.reward).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)
        next_state_additional_batch = torch.stack(transition_batch.next_state_additional).to(device)

        nn_output_vectors = self.policy_net(state_batch, state_additional_batch)
        state_action_values = []
        for i in range(0, len(action_batch)):
            state_action_values.append(nn_output_vectors[i][0][action_batch[i][0].item()])  # TODO use gather instead
        state_action_values = torch.stack(state_action_values)  # list of tensors to tensor

        # next_state_values = torch.zeros(len(non_final_next_states), device=device)
        # next_state_Values = torch.zeros(self.BATCH_SIZE, device=device)
        targ_output_planner = self.target_net(next_state_batch, next_state_additional_batch)
        next_state_values = torch.max(targ_output_planner, dim=1)
        # for i in range(len(targ_output_planner)):
        #     next_state_values[i] = torch.max(targ_output_planner[i])

        # next_state_Values[non_final_mask] = next_state_values

        # Compute the expected Q values
        expected_state_action_values = (1 - done_batch)(next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.MSELoss()
        loss_p = criterion(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss_p.backward()
        self.optimizer.step()

    def select_action(self, state, state_additional, steps_done):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was found
                planner_vector = self.policy_net(state, state_additional)
                return planner_vector.max(1)[1].view(1, 1)
        else:
            actionNo = torch.tensor([[random.randrange(self.n_actions)]], device=device)
            return actionNo

    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


