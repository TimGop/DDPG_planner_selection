import math
import os
import random

import torch
from torchvision.io import read_image
from typing import Optional
import gym
import numpy as np


class PortfolioEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, image_dir, max_time, func_reward_planner, func_reward_time, t_SD, p_SD, nb_planners,
                 omnicron, Theta, Epsilon,
                 time_per_ep):
        super(PortfolioEnvironment, self).__init__()
        # General Information for this Environment
        self.df = df
        self.image_dir = image_dir
        self.nb_planners = nb_planners
        self.max_time = max_time
        self.func_reward_planner = func_reward_planner
        self.func_reward_time = func_reward_time
        self.time_standard_dev = t_SD
        self.planner_standard_dev = p_SD
        self.omnicron = omnicron
        self.Theta = Theta
        self.Epsilon = Epsilon
        self.time_per_ep = time_per_ep

        # Current State of this Environment
        self.num_steps = 0
        self.best_planner_time = (df.min(axis=1))[0]
        self.last_action_number = None
        self.task_idx = -1
        self.task_img = None
        self.time_left = -1
        self.max_time_executed = np.array([0.] * self.nb_planners)
        self.consecutive_time_running = np.array([0.] * self.nb_planners)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.num_steps = 0
        self.task_idx = math.floor(random.random() * len(self.df))
        self.best_planner_time = self.df.min(axis=1)[self.task_idx]
        task_name = self.df.iloc[self.task_idx][0]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        self.time_left = self.max_time
        self.max_time_executed[:] = 0
        self.consecutive_time_running[:] = 0
        return self.get_Observation(), {}

    def reset_testMode(self, idx):
        self.num_steps = 0
        self.task_idx = idx
        self.best_planner_time = self.df.min(axis=1)[self.task_idx]
        task_name = self.df.iloc[self.task_idx][0]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        self.time_left = self.max_time
        self.max_time_executed[:] = 0
        self.consecutive_time_running[:] = 0
        # returns current state as a tuple
        return self.get_Observation(), {}

    def get_Observation(self):
        obs = {
            "task_img": self.task_img,
            "task_additional": np.append(np.concatenate((self.consecutive_time_running, self.max_time_executed)),
                                         self.time_left)
        }
        return obs

    def step(self, action):
        action_time = self.time_left if self.time_left < action[-1] else action[-1]  # clip time
        self.num_steps += 1
        # assert self.action_space.contains(action), "action space doesnt contain action..."
        assert self.task_img is not None, "Call reset before using step method..."
        final_state = False
        time_limit = False
        # action contains planner values and one time output at the end
        action_number = int(action[0])
        if self.last_action_number == action_number:  # to update consecutive times below
            same_action = True
        else:
            same_action = False
        rewardVal_planner, done_planner = self.func_reward_planner(self.task_idx, action_number,
                                                                   self.consecutive_time_running[action_number],
                                                                   self.time_left,
                                                                   self.df,
                                                                   self.omnicron, self.Theta, self.Epsilon)
        rewardVal_time, done_time = self.func_reward_time(self.task_idx, action_number, action[-1],
                                                          self.consecutive_time_running[action_number], self.time_left,
                                                          self.df,
                                                          self.omnicron, self.Theta, self.Epsilon, self.time_per_ep)
        done = done_planner and done_time

        self.time_left -= action_time
        current_planner_time = self.df.iloc[self.task_idx][action_number + 1]  # +1 because first col ist task title
        time_up = ((self.time_left - self.best_planner_time) <= 0) and \
                  ((self.time_left + self.consecutive_time_running[action_number]) - current_planner_time <= 0)
        if same_action:
            self.consecutive_time_running[action_number] += action_time
        else:
            self.consecutive_time_running = np.zeros((self.nb_planners,))
            self.consecutive_time_running[action_number] = self.consecutive_time_running[action_number] + action_time

        if self.max_time_executed[action_number] < self.consecutive_time_running[action_number]:
            self.max_time_executed[action_number] = self.consecutive_time_running[action_number]
        self.last_action_number = action_number

        leq_zero_action = (action_time <= 0)
        # (not going to fix problem)
        if done or time_up or leq_zero_action:
            if done:
                final_state = True
                self.task_img = None
            else:
                time_limit = True
                self.task_img = None

        if self.num_steps == 10 and not done:
            time_limit = True
            rewardVal_planner = -1
            rewardVal_time = -1
        return self.get_Observation(), (rewardVal_planner, rewardVal_time), final_state, time_limit, {}

    def get_time_noise(self):
        return torch.normal(torch.zeros((1,)), torch.full((1,), self.time_standard_dev))

    def render(self, mode='human', close=False):
        print(f"Tas: {self.task_idx}")
        print(f"Time Left: {self.time_left}")
        print(f"Max Time Executed: {self.max_time_executed}")
        print(f"Consecutive Time Running: {self.consecutive_time_running}")
