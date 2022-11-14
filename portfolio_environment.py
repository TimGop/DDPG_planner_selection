# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

import math
import os
import random

import gym
from gym import spaces
import numpy as np


class PortfolioEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, df, image_dir, max_time, func_reward):
    super(PortfolioEnvironment, self).__init__()
    # General Information for this Environment
    self.df = df
    self.image_dir = image_dir
    self.nb_planners = 17 # TODO: extract from df
    self.max_time = max_time
    self.func_reward = func_reward

    # Action and Observation Space
    # One "action" per planner + time
    self.action_space = spaces.Box(
      low=np.array([0] * self.nb_planners + [0]),
      high=np.array([1] * self.nb_planners + [self.max_time])
    )    # Example for using image as input:
    self.observation_space = spaces.Dict({
      "task_img":  spaces.Box(low=0, high=1, shape=(X, Y)),
    }) # TODO: add the other elements of a current state to the observation space
    spaces.Box()  # TODO: Based on current state below
    # Patrick: Observation description for AN image. Not for our images and not
    # for the other inputs
    # self.observation_space =

    # Current State of this Environment
    self.task_idx = -1
    self.task_img = None
    self.time_left = -1
    self.max_time_executed = np.array([0] * self.nb_planners)
    self.consecutive_time_running = np.array([0] * self.nb_planners)

  def reset(self):
    self.task_idx = math.floor(random.random() * len(self.df))
    task_name = self.df.iloc[self.task_idx][0]
    self.task_img = np.ascontiguousarray(read_image(os.path.join(
      self.image_dir, task_name+ '-bolded-cs.png')), dtype=np.float32) / 255
    self.time_left = self.max_time
    self.max_time_executed[:] = 0
    self.consecutive_time_running[:] = 0
    return self._next_observation()

  def _next_observation(self):
    # TODO: Adapt from example
    # Get the data points for the last 5 days and scale to between 0-1
    frame = np.array([
      self.df.loc[self.current_step: self.current_step +
                                     5, 'Open'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                                     5, 'High'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                                     5, 'Low'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                                     5, 'Close'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                                     5, 'Volume'].values / MAX_NUM_SHARES,
    ])  # Append additional data and scale each value to between 0-1
    obs = np.append(frame, [[
      self.balance / MAX_ACCOUNT_BALANCE,
      self.max_net_worth / MAX_ACCOUNT_BALANCE,
      self.shares_held / MAX_NUM_SHARES,
      self.cost_basis / MAX_SHARE_PRICE,
      self.total_shares_sold / MAX_NUM_SHARES,
      self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
    ]], axis=0)
    return obs


  def step(self, action):
    # TODO: Adapt
    # Execute one time step within the environment
    self._take_action(action)
    self.current_step += 1
    if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
      self.current_step = 0
      delay_modifier = (self.current_step / MAX_STEPS)

    reward = self.balance * delay_modifier
    done = self.net_worth <= 0
    obs = self._next_observation()
    return obs, reward, done, {}

  def render(self, mode='human', close=False):
    print(f"Tas: {self.task_idx}")
    print(f"Time Left: {self.time_left}")
    print(f"Max Time Executed: {self.max_time_executed}")
    print(f"Consecutive Time Running: {self.consecutive_time_running}")
