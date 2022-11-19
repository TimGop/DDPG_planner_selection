import random
from typing import Optional
import gym
from gym import spaces


def initialize_pos():
    left_or_right = random.random() > 0.5
    ret = random.random() * 0.7
    if left_or_right:
        ret += 0.8
    return ret


class lineEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(lineEnvironment, self).__init__()
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
        self.pos = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.pos = initialize_pos()
        return self.pos, {}

    def step(self, action):
        self.pos = self.get_New_pos(action)

        if 0.8 >= self.pos >= 0.7:
            rewardVal = 1.0
            return self.pos, rewardVal, True, False, {}
        else:
            rewardVal = 0.0
            return self.pos, rewardVal, False, False, {}

    def get_New_pos(self, action):
        pos = self.pos + action
        # clip position to domain if necessary
        if pos < self.obs_low:
            pos = self.obs_low
        elif pos > self.obs_high:
            pos = self.obs_high
        return pos

    def render(self, mode='human', close=False):
        print(f"position on line between 0 and 1.5 is: {self.pos}")
