# https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e

import math
import os
import random
from torchvision.io import read_image
from typing import Optional
import gym
from gym import spaces
import numpy as np


class PortfolioEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, image_dir, max_time, func_reward, X, Y):
        super(PortfolioEnvironment, self).__init__()
        # General Information for this Environment
        self.df = df
        self.image_dir = image_dir
        self.nb_planners = 17  # len(df.columns)
        self.max_time = max_time
        self.func_reward = func_reward

        # Action and Observation Space
        # One "action" per planner + time
        self.action_space = spaces.Box(
            low=np.array([0.] * self.nb_planners + [0.]),
            high=np.array([1.] * self.nb_planners + [float(self.max_time)])
        )  # Example for using image as input:
        self.observation_space = spaces.Dict({
            "task_img": spaces.Box(low=0, high=1, shape=(X, Y)),
            "task_additional": spaces.Box(low=np.array([0] * 2 * self.nb_planners + [0]),
                                          high=np.array(np.array([1800] * 2 * self.nb_planners + [0])))
        })
        # Patrick: Observation description for AN image. Not for our images and not
        # for the other inputs
        # self.observation_space =

        # Current State of this Environment
        self.best_planner_time = (df.min(axis=1))[0]
        self.last_action_number = None
        self.task_idx = -1
        self.task_img = None
        self.time_left = -1
        self.max_time_executed = np.array([0.] * self.nb_planners)
        self.consecutive_time_running = np.array([0.] * self.nb_planners)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.task_idx = math.floor(random.random() * len(self.df))
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
        # assert self.action_space.contains(action), "action space doesnt contain action..."
        assert self.task_img is not None, "Call reset before using step method..."
        final_state = False
        time_limit = False
        action_number = np.argmax(action[:17])
        if self.last_action_number == action_number:  # to update consecutive times below
            same_action = True
        else:
            same_action = False
        rewardVal, done = self.func_reward(self.task_idx, action_number, action[-1],
                                           self.consecutive_time_running[action_number], self.time_left, self.df)
        self.time_left -= action[-1]
        leq_zero_action = (action[-1] <= 0)
        time_up = ((self.time_left - self.best_planner_time) <= 0)
        if same_action:
            self.consecutive_time_running[action_number] += action[-1]
        else:
            self.consecutive_time_running = np.zeros((self.nb_planners,))
            self.consecutive_time_running[action_number] = self.consecutive_time_running[action_number] + action[-1]
        if self.max_time_executed[action_number] < self.consecutive_time_running[action_number]:
            self.max_time_executed[action_number] = self.consecutive_time_running[action_number]
        self.last_action_number = action_number
        if done or time_up or leq_zero_action:
            if done:
                final_state = True
                self.task_img = None
            else:
                time_limit = True
                self.task_img = None
        return self.get_Observation(), rewardVal, final_state, time_limit, {}

    def render(self, mode='human', close=False):
        print(f"Tas: {self.task_idx}")
        print(f"Time Left: {self.time_left}")
        print(f"Max Time Executed: {self.max_time_executed}")
        print(f"Consecutive Time Running: {self.consecutive_time_running}")