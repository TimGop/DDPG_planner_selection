import portfolio_environment
import pandas as p
from DDPG_reward import reward
import numpy as np

testSet = p.read_csv("IPC-image-data-master/problem_splits/testing.csv")
taskFolderLoc = "IPC-image-data-master/grounded/"

env = portfolio_environment.PortfolioEnvironment(testSet, taskFolderLoc, reward)

num_correct = 0
for i in range(1000):
    env.reset()
    action = np.array([1]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0]+[0])
    _, rewardVal, final_state, _, _ = env.step(action)
    print(final_state)
    print(rewardVal)
    if final_state:
        num_correct += 1

print("percentage correct", num_correct/1000)

