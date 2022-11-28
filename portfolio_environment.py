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

    def __init__(self, df, image_dir, func_reward, time_per_ep=1800, omnicron=10, Theta=10, Epsilon=1):
        super(PortfolioEnvironment, self).__init__()
        # General Information for this Environment
        self.time_per_ep = time_per_ep
        self.omnicron = omnicron
        self.Theta = Theta
        self.Epsilon = Epsilon
        self.filenames = df["filename"]
        self.df = df.drop("filename", axis="columns")
        self.image_dir = image_dir
        self.func_reward = func_reward
        self.best_planner_times = (self.df.min(axis=1))

        # Current State of this Environment
        self.best_planner_time = None
        self.task_idx = -1
        self.task_img = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.task_idx = math.floor(random.random() * len(self.df))
        self.best_planner_time = self.best_planner_times[self.task_idx]
        task_name = self.filenames.iloc[self.task_idx]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        return self.task_img, {}

    def reset_testMode(self, idx):
        self.task_idx = idx
        self.best_planner_time = self.df.min(axis=1)[self.task_idx]
        task_name = self.filenames.iloc[self.task_idx]
        self.task_img = np.ascontiguousarray(read_image(os.path.join(
            self.image_dir, task_name + '-bolded-cs.png')), dtype=np.float32) / 255
        return self.task_img, {}

    def step(self, action):
        assert self.task_img is not None, "Call reset before using step method..."
        action_number = np.argmax(action[:17])
        rewardVal, done = self.func_reward(self.task_idx, action_number, self.df, time_per_ep=self.time_per_ep,
                                           ominicron=self.omnicron, Theta=self.Theta, Epsilon=self.Epsilon)
        if done:
            self.task_img = None
        return self.task_img, rewardVal, done, True, {}

    @staticmethod
    def get_planner_noise():
        return torch.normal(torch.zeros((1, 17)), torch.full((1, 17), 1.0))

    def render(self, mode='human', close=False):
        print(f"Tas: {self.task_idx}")
