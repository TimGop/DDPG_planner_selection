from DDPG_nets import Actor, Critic
import torch
import torch.nn.functional as F
from torch.optim import Adam


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class DDPG(object):
    def __init__(self, gamma, tau):
        self.gamma = gamma
        self.tau = tau
        # Define the actor
        self.actor = Actor(128, 128, 17).to(device)
        self.actor_target = Actor(128, 128, 17).to(device)

        # Define the critic
        self.critic = Critic(128, 128).to(device)
        self.critic_target = Critic(128, 128).to(device)

        self.critic_target.eval()  # removes dropout etc. for evaluation purposes
        self.actor_target.eval()  # removes dropout etc. for evaluation purposes

        self.actor_optimizer = Adam(self.actor.parameters())  # optimizer for the actor network
        self.critic_optimizer = Adam(self.critic.parameters())  # optimizer for the critic network

        hard_update(self.critic_target, self.critic)  # make sure _ and target have same weights
        hard_update(self.actor_target, self.actor)  # make sure _ and target have same weights

    def act(self, state):
        # TODO do we need action noise or rand decisions scaled down over time?
        self.actor.eval()
        ans = self.actor(state)
        self.actor.train()
        ans = ans.data

        # # During training we add noise for exploration
        # if action_noise is not None:
        #     noise = torch.Tensor(action_noise.noise()).to(device)
        #     mu += noise

        # Clip the output according to the action space of the env 0-1800??? --> no bad idea
        # ans = ans.clamp(0, 1800)
        return ans

    def update(self, transition_batch):
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        action_batch = torch.cat(transition_batch.action).to(device)
        reward_batch = torch.cat(transition_batch.reward).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # TODO: Clipping the expected values here?  ---> why clip expected values???
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))  # TODO why minus in front???
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def set_eval(self):  # TODO find when used
        # Sets the model in evaluation mode
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):  # TODO find when used
        # Sets the model in training mode
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()
