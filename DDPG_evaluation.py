import torch
import random
import numpy as np
import pandas as p

import portfolio_environment
from DDPG_reward import reward
import matplotlib.pyplot as plt
from Replay_Memory_and_utils import resize

testSet = p.read_csv("IPC-image-data-master/problem_splits/testing.csv")
taskFolderLoc = "IPC-image-data-master/lifted/"
env = portfolio_environment.PortfolioEnvironment(testSet, taskFolderLoc, reward)


def randAction(n_actions):
    return torch.tensor([[random.random() for _ in range(n_actions)]], dtype=torch.float32).softmax(dim=1)


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, agent, randAverageReward, rand_bool=False,
                    n_actions=17):
    print("start testing...")
    agent.set_eval()
    num_of_tests = len(testSet)
    rewardTotal = 0
    number_of_passes = 0
    number_correct = 0
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        obs, _ = env.reset_testMode(task_i_idx)
        task_img = torch.from_numpy(obs)
        state = resize(task_img).unsqueeze(0)
        episode_end = False
        i = 0
        while not episode_end:
            i += 1
            if not rand_bool:
                action = agent.get_action(state)
            else:
                action = randAction(n_actions)
            print(torch.argmax(action).item())
            action = np.array(action.detach().squeeze(0))
            obs, rewardVal, final_state, episode_end, _ = env.step(action)
            rewardTotal += rewardVal
            number_of_passes += 1
            if final_state:
                number_correct += 1

    averageRewards.append((rewardTotal / number_of_passes))
    print("percentage correct=" + str((number_correct / num_of_tests) * 100) + "%")
    if not rand_bool:
        print("Episode", episodeNumbers)
        print("Avg Reward", averageRewards)
        if episodeNumbers.__len__() > 1:
            plt.plot(episodeNumbers, averageRewards, color='g', label="policy network")
        else:
            plt.axhline(y=averageRewards[0], color='g', label="policy network")
        plt.axhline(y=randAverageReward, label="random action baseline")
        plt.xlabel('number of episodes')
        plt.ylabel('average reward')
        plt.title('average rewards while testing DDPG:')
        plt.legend()
        plt.show()
    agent.set_train()
    print("finish testing...")
    return episodeNumbers, averageRewards
