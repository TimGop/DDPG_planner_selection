import torch
import random
import numpy as np
import pandas as p
from torchvision.io import read_image
from DDPG_reward import reward
import matplotlib.pyplot as plt
from Replay_Memory_and_utils import resize

testSet = p.read_csv("C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/testing.csv")
taskFolderLoc = "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/grounded/"


def randAction(timeLeft, n_actions):
    # action = torch.tensor([[random.randrange(n_actions), random.random() * timeLeft]])
    # return action
    return torch.tensor([[random.random() for _ in range(n_actions)]]).softmax(dim=1), \
           torch.tensor([random.random() * timeLeft])


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, agent, randAverageReward, rand_bool=False,
                    time_per_ep=1800, n_actions=17):
    print("start testing...")
    agent.set_eval()
    minTimeReq_best_planner_list_test = testSet.min(axis=1)
    num_of_tests = len(testSet)
    rewardTotal = 0
    number_of_passes = 0
    number_correct = 0
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        # print(task_i_idx)
        e_time_left_ep = time_per_ep  # 1800
        e_maxConsecExecuted = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        e_currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
        e_current_task_index = task_i_idx
        e_currentTaskName = testSet.iloc[e_current_task_index][0]
        e_currentTaskLoc = taskFolderLoc + e_currentTaskName + '-bolded-cs.png'
        e_img = read_image(e_currentTaskLoc)
        e_img = np.ascontiguousarray(e_img, dtype=np.float32) / 255
        e_img = torch.from_numpy(e_img)
        e_img = resize(e_img).unsqueeze(0)
        state = e_img
        state_additional = torch.cat((e_maxConsecExecuted, e_currentlyExecuting, torch.tensor([e_time_left_ep])))
        minTimeReq_best_planner_testSet = minTimeReq_best_planner_list_test[e_current_task_index]
        prevActionIdx = None
        while 0 <= e_time_left_ep - minTimeReq_best_planner_testSet:
            if not rand_bool:
                action = agent.get_action(state, state_additional)
            else:
                action = randAction(e_time_left_ep, n_actions)
            action_idx = torch.argmax(action[0]).item()  # conversion to int
            action_t = action[1][0].item()
            currReward = reward(e_current_task_index, action_idx, action_t, e_time_left_ep, testSet)[0]
            rewardTotal += currReward
            number_of_passes += 1
            # actionNo.item+1 because first column is name
            if action_t == 0:
                # to avoid infinite loops
                break
            elif testSet.iloc[e_current_task_index][action_idx + 1] > action_t:
                # action hasnt led to goal continue
                if prevActionIdx is action_idx:
                    e_currentlyExecuting[action_idx] += action_t
                else:
                    e_currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        dtype=torch.float)
                    e_currentlyExecuting[action_idx] += action_t
                e_time_left_ep -= action_t
                if e_currentlyExecuting[action_idx] > e_maxConsecExecuted[action_idx]:
                    e_maxConsecExecuted[action_idx] = e_currentlyExecuting[action_idx]
                state = e_img
                state_additional = torch.cat((e_maxConsecExecuted, e_currentlyExecuting, torch.tensor([e_time_left_ep])))
            else:
                number_correct += 1
                # action leads to goal i.e. done
                break
            prevActionIdx = action_idx
    averageRewards.append((rewardTotal / num_of_tests).item())
    if not rand_bool:
        print(episodeNumbers)
        print(averageRewards)
        print("percentage correct=" + str((number_correct / num_of_tests) * 100) + "%")
        print("average number of passes per task=" + str(number_of_passes / num_of_tests))
        # plotting average reward development throughout test set
        if episodeNumbers.__len__() > 1:
            plt.plot(episodeNumbers, averageRewards, color='g', label="policy network")
        else:
            plt.axhline(y=averageRewards[0], color='g', label="policy network")
        plt.axhline(y=randAverageReward, label="random action baseline")
        plt.xlabel('number of episodes')
        plt.ylabel('average reward')
        plt.title('average rewards while testing DQN:')
        plt.legend()
        plt.show()
    agent.set_train()
    print("finish testing...")
    return episodeNumbers, averageRewards
