from DDPG_nets import Actor, Critic
import torch
import torch.nn.functional as F
from torch.optim import Adam

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):
    def __init__(self, gamma, tau, h, w, num_planners, env):
        self.gamma = gamma
        self.tau = tau

        self.env = env

        # Define the actor
        self.actor = Actor(h, w, num_planners).to(device)
        self.actor_target = Actor(h, w, num_planners).to(device)

        # Define the critic
        self.critic = Critic(h, w, NumAdditionalArgsLinLayer=num_planners).to(device)
        self.critic_target = Critic(h, w, NumAdditionalArgsLinLayer=num_planners).to(device)

        self.critic_target.eval()  # removes dropout etc. for evaluation purposes
        self.actor_target.eval()  # removes dropout etc. for evaluation purposes

        self.actor_optimizer = Adam(self.actor.parameters())  # optimizer for actor net
        self.critic_optimizer = Adam(self.critic.parameters())  # optimizer for critic net

        hard_update(self.critic_target, self.critic)  # make sure _ and target have same weights
        hard_update(self.actor_target, self.actor)  # make sure _ and target have same weights

    def get_action(self, select_action_State):
        self.actor.eval()
        action = self.actor(select_action_State)
        self.actor.train()
        return action

    def act(self, select_action_State):
        # actions are the softmaxed values for the 17 planners
        with torch.no_grad():
            actions = self.get_action(select_action_State)
            actions += self.env.get_planner_noise()
            return actions

    def update(self, transition_batch):
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        action_batch = torch.stack(transition_batch.action).to(device)
        action_batch = action_batch.squeeze(dim=1)
        reward_batch = torch.stack(transition_batch.reward).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        # print("Critic expected values: ", expected_values)
        # print("Critic values: ", state_action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        # print("state_action_batch: ", state_action_batch)
        # print("expected_values: ", expected_values.detach())
        # print("val_loss vec: ", expected_values.detach()-state_action_batch)
        # print("val loss: ", value_loss)
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        # minus in front below because comes from a q-value
        # print("next_time_target: ", next_time_batch)
        # print("state_time_policy: ", state_time)
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
        # print("policy loss from critic for actor: ", policy_loss)
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def set_eval(self):
        # Sets the model in evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        # Sets the model in training mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
