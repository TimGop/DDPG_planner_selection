import math
import random
from DDPG_nets import Actor, Critic
import torch
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200  # try different values?

n_actions = 17
steps_done = 0


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):
    def __init__(self, gamma, tau, h, w):
        self.gamma = gamma
        self.tau = tau

        # Define the actor
        self.actor = Actor(h, w, 17).to(device)
        self.actor_target = Actor(h, w, 17).to(device)

        # Define the critic
        self.critic = Critic(h, w).to(device)
        self.critic_target = Critic(h, w).to(device)

        self.critic_target.eval()  # removes dropout etc. for evaluation purposes
        self.actor_target.eval()  # removes dropout etc. for evaluation purposes

        self.actor_optimizer = Adam(self.actor.parameters())  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters())  # optimizer for the critic network

        hard_update(self.critic_target, self.critic)  # make sure _ and target have same weights
        hard_update(self.actor_target, self.actor)  # make sure _ and target have same weights

    def get_action(self, select_action_State, select_action_State_additional):
        self.actor.eval()
        action = self.actor(select_action_State, torch.unsqueeze(select_action_State_additional, dim=0))
        self.actor.train()
        return action

    def act(self, select_action_State, select_action_State_additional):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.get_action(select_action_State, select_action_State_additional)
        else:
            return torch.tensor([[random.random() for _ in range(n_actions)]], device=device).softmax(dim=1),\
                   torch.tensor([random.random() * select_action_State_additional[-1]],
                                device=device)

    def update(self, transition_batch):
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        state_additional_batch = torch.stack(transition_batch.state_additional).to(device)
        action_batch = torch.stack(transition_batch.action).to(device)
        time_batch = torch.stack(transition_batch.time).to(device)
        reward_batch = torch.cat(transition_batch.reward).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)
        next_state_additional_batch = torch.stack(transition_batch.next_state_additional).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch, next_time_batch = self.actor_target(next_state_batch, next_state_additional_batch)
        print("next_time", next_time_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_state_additional_batch,
                                                      next_action_batch.detach(), next_time_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        #print(action_batch.shape)
        state_action_batch = self.critic(state_batch, state_additional_batch, action_batch, time_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        # minus in front below because comes from a q-value
        state_actions, state_time = self.actor(state_batch, state_additional_batch)
        print("state_time", state_time)
        #print(temp.shape)
        policy_loss = -self.critic(state_batch, state_additional_batch, state_actions, state_time)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def set_eval(self):  # sets networks to evaluation mode (faster)
        # Sets the model in evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):  # sets networks to training mode (needed for learning but slower)
        # Sets the model in training mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
