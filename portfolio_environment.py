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

    def __init__(self, df, image_dir, func_reward):
        super(PortfolioEnvironment, self).__init__()

        # General Information for this Environment
        self.df = df
        self.image_dir = image_dir
        self.nb_planners = len(df.columns) - 1
        self.func_reward = func_reward

        # Current State of this Environment
        self.best_planner_time = (df.min(axis=1))[0]
        self.task_idx = -1
        self.task_img = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.task_idx = math.floor(random.random() * len(self.df))
        self.best_planner_time = self.df.min(axis=1)[self.task_idx]
        task_name = self.df.iloc[self.task_idx][0]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        return self.task_img, {}

    def reset_testMode(self, idx):
        self.task_idx = idx
        self.best_planner_time = self.df.min(axis=1)[self.task_idx]
        task_name = self.df.iloc[self.task_idx][0]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        return self.task_img, {}

    def step(self, action):
        assert self.task_img is not None, "Call reset before using step method..."
        final_state = False
        action_number = np.argmax(action[:17])
        rewardVal, done = self.func_reward(self.task_idx, action_number, self.df)
        if done:
            final_state = True
            self.task_img = None
        return self.task_img, rewardVal, final_state, True, {}

    @staticmethod
    def get_planner_noise():
        return torch.normal(torch.zeros((1, 17)), torch.full((1, 17), 0.3))

    def render(self, mode='human', close=False):
        print(f"Tas: {self.task_idx}")

