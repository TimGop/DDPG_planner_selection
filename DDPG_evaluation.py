# TODO rewrite
import torch
import numpy as np
from torchvision.io import read_image


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, randAverageReward, rand_bool=False,
                    time_per_ep=1800):
    print("start testing...")

    minTimeReq_best_planner_list_test = testSet.min(axis=1)
    num_of_tests = len(testSet)
    rewardTotal = 0
    number_of_passes = 0
    number_correct = 0
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        print(task_i_idx)
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
        state = (e_img, e_current_task_index, e_maxConsecExecuted, e_currentlyExecuting, torch.tensor([e_time_left_ep]))
        minTimeReq_best_planner_testSet = minTimeReq_best_planner_list_test[e_current_task_index]
        prevActionIdx = None
        while 0 <= e_time_left_ep - minTimeReq_best_planner_testSet:
            if not rand_bool:
                actionVector, Action_t = policy_net(state)
            else:
                actionVector, Action_t = randAction(e_time_left_ep)
            # print(Action_t)
            action_idx = actionVector.max(1)[1].view(1, 1)
            currReward = reward(e_current_task_index, action_idx.item(), Action_t, e_time_left_ep, testSet)[0]
            rewardTotal += currReward
            number_of_passes += 1
            # actionNo.item+1 because first column is name
            if testSet.iloc[e_current_task_index][action_idx.item() + 1] > Action_t:
                # action hasnt led to goal continue
                if prevActionIdx is action_idx:
                    e_currentlyExecuting[action_idx] += Action_t.item()
                else:
                    e_currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                        dtype=torch.float)
                    e_currentlyExecuting[action_idx] += Action_t.item()
                e_time_left_ep -= Action_t
                if e_currentlyExecuting[action_idx] > e_maxConsecExecuted[action_idx]:
                    e_maxConsecExecuted[action_idx] = e_currentlyExecuting[action_idx]
                state = (
                    e_img, e_current_task_index, e_maxConsecExecuted, e_currentlyExecuting,
                    torch.tensor([e_time_left_ep]))
            else:
                number_correct += 1
                # action leads to goal
                break
            prevActionIdx = action_idx
    averageRewards.append((rewardTotal / number_of_passes).item())
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
    return episodeNumbers, averageRewards