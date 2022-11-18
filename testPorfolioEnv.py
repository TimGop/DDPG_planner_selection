import numpy as np

import portfolio_environment
import pandas as p
from DDPG_reward import reward2

testSet = p.read_csv("IPC-image-data-master/problem_splits/testing.csv")
taskFolderLoc = "IPC-image-data-master/grounded/"
env = portfolio_environment.PortfolioEnvironment(testSet, taskFolderLoc, 1800, reward2, 128, 128)
print(env.reset())
print()
print()
action = np.array([1] + [0] * 16 + [20])
print("action: ", action)
print("task_name: ", testSet.iloc[env.task_idx][0])
print("best time: ", env.best_planner_time)
print()
print()
print(env.step(action=action))
print()
print()
action = np.array([0] + [1] + [0] * 15 + [1700])
print("action: ", action)
print()
print()
print(env.step(action=action))
print()
print()
action = np.array([0] + [1] + [0] * 15 + [50])
print("action: ", action)
print()
print()
print(env.step(action=action))

