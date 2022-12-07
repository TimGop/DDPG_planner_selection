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
    def __init__(self, gamma, tau, h, w, env, num_planners):
        self.gamma = gamma
        self.tau = tau
        self.env = env

        # Define the actor
        self.actor = Actor(h, w, num_planners=num_planners).to(device)
        # nn.init.kaiming_normal_(self.actor.headTime.weight, nonlinearity='relu')
        # nn.init.constant_(self.actor.headTime.bias, 0)
        # self.actor.apply(init_weights)
        # print("conv2d init weights: ", self.actor.conv2d.weight)
        self.actor_target = Actor(h, w, num_planners=num_planners).to(device)

        # Define the critic
        self.critic = Critic(h, w, num_planners=num_planners).to(device)
        # nn.init.xavier_normal_(self.critic.headQ.weight)
        # nn.init.constant_(self.critic.headQ.bias, 0)
        # self.critic.apply(init_weights)
        self.critic_target = Critic(h, w, num_planners=num_planners).to(device)

        self.critic_target.eval()  # removes dropout etc. for evaluation purposes
        self.actor_target.eval()  # removes dropout etc. for evaluation purposes

        self.actor_optimizer = Adam(self.actor.parameters())  # optimizer for actor net
        self.critic_optimizer = Adam(self.critic.parameters())  # optimizer for critic net

        hard_update(self.critic_target, self.critic)  # make sure _ and target have same weights
        hard_update(self.actor_target, self.actor)  # make sure _ and target have same weights

    def get_action(self, select_action_State, select_action_State_additional, discrete_action):
        self.actor.eval()
        action = self.actor(select_action_State, torch.unsqueeze(select_action_State_additional, dim=0),
                            discrete_action)
        self.actor.train()
        return action

    def act(self, select_action_State, select_action_State_additional, discrete_action):
        with torch.no_grad():
            action = self.get_action(select_action_State, select_action_State_additional, discrete_action)
            action += self.env.get_time_noise()
            return action

    def update(self, transition_batch, agent_dqn):
        self.set_train()
        state_batch = torch.cat(transition_batch.state).to(device)
        state_additional_batch = torch.stack(transition_batch.state_additional).to(device)
        action_batch = torch.stack(transition_batch.action).to(device)
        time_batch = torch.stack(transition_batch.time).to(device)
        reward_batch = torch.cat(transition_batch.reward_time).to(device)
        done_batch = torch.cat(transition_batch.done).to(device)
        next_state_batch = torch.cat(transition_batch.next_state).to(device)
        next_state_additional_batch = torch.stack(transition_batch.next_state_additional).to(device)

        # Get the actions and the state values to compute the targets
        next_action_batch = torch.argmax(agent_dqn.target_net(next_state_batch, next_state_additional_batch),
                                         dim=1).detach().unsqueeze(1)

        next_time_batch = self.actor_target(next_state_batch, next_state_additional_batch, next_action_batch)

        next_state_action_values = self.critic_target(next_state_batch, next_state_additional_batch,
                                                      next_action_batch.detach(), next_time_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, state_additional_batch, action_batch, time_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)  # gradient clipping
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        state_time = self.actor(state_batch, state_additional_batch, action_batch)
        # minus in front below because comes from a q-value
        policy_loss = -self.critic(state_batch, state_additional_batch, action_batch, state_time)
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)  # gradient clipping
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

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
