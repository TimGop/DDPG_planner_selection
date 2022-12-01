import random
from typing import Optional
import gym
from gym import spaces
import torch


def initialize_pos():
    left_or_right = random.random() > 0.5
    ret = random.random() * 0.725
    if left_or_right:
        ret += 0.775
    return ret


class lineEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(lineEnvironment, self).__init__()

        self.finished = False
        self.max_game_length = 20
        self.action_space = spaces.Box(
            low=-0.3,
            high=0.3
        )
        self.obs_low = 0.0
        self.obs_high = 1.5
        self.observation_space = spaces.Box(
            low=self.obs_low,
            high=self.obs_high
        )
        self.round = 0
        self.pos = 0.0
        self.history = []

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.pos = initialize_pos()
        self.finished = False
        self.history = []
        self.round = 0
        return self.pos, {}

    def step(self, action):
        assert not self.finished, "reset() after reaching final state..."
        self.round += 1
        self.pos, clipped = self.get_New_pos(action)
        out_of_range_penalty = 0
        if clipped:
            out_of_range_penalty = -0.2
        self.history.append((action, self.pos))
        solved = False
        if 0.775 >= self.pos >= 0.725:
            solved_reward = 2.5
            solved = True
            self.finished = True
        elif self.round > self.max_game_length:
            self.finished = True
            solved_reward = -1
        else:
            solved_reward = -0.1
        return self.pos, solved_reward + out_of_range_penalty, solved, self.finished, {}

    def get_New_pos(self, action):
        pos = self.pos + action
        # clip position to domain if necessary
        clip = False
        if pos < self.obs_low:
            pos = self.obs_low
            clip = True
        elif pos > self.obs_high:
            pos = self.obs_high
            clip = True
        return pos, clip

    def render(self, mode='human', close=False):
        print("ACTIONS", [x[0] for x in self.history])
        print("STATE", [x[1] for x in self.history])
        print("LEN EP", len(self.history))
        # print(f"position on line between 0 and 1.5 is: {self.pos}")

    @staticmethod
    def get_noise():
        return torch.normal(torch.tensor(0.0), torch.tensor(0.2))
