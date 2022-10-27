import math
import time

import numpy as np
import pandas as p
import torch
import torch.nn as nn
import torch.optim as opt
import random
from collections import namedtuple, deque
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
trainingSet = p.read_csv(
    "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/training.csv")

taskFolderLoc = "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/grounded/"

imageHeight = 3  # must be 128 or smaller
imageWidth = 3  # must be 128 or smaller


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        numOutputChannelsConvLayer = 128
        self.conv2d = nn.Conv2d(1, numOutputChannelsConvLayer, kernel_size=(2, 2), stride=(1, 1))
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        NumAdditionalArgsLinLayer = 35
        # NumAdditionalArgsLinLayer: For each planner currently executing and max consecutively executing (2*17)
        #                            plus 1 more for time remaining in episode --> (2*17+1=35)
        linear_input_size = ((h - 1) * (w - 1) * numOutputChannelsConvLayer) + NumAdditionalArgsLinLayer
        self.headPlanner = nn.Linear(linear_input_size, outputs - 1)
        self.headTime = nn.Linear(linear_input_size, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, f_state):  # implementation with single input arg seems to be faster at the moment
        # TODO test and plot after every network update while training(calculate randomBaseline prior)
        #  + change % correct
        # TODO improve reward function
        # TODO sequential seperate encoding python file
        # TODO improve efficiency of forward (look at reinforcement-q-learning.py in forward --> HOW!?!?!?!
        # DQN_planner_choice and DQN old code compare speeds in forward() --> bottleneck in layers???(cant control)
        # TODO improve nn learning for time selection (time slots alloc. by nn too small!!!)
        if type(f_state) is list:
            # f_state is a batch of states
            ret_list = []
            t_list = []
            tic1 = 0
            tic2 = 0
            tic3 = 0
            tic4 = 0
            for i_state in f_state:
                lastTime = time.time()
                x = i_state[0]  # img
                x = x.to(device)

                t = time.time()
                tic1 += t - lastTime
                lastTime = t
                x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
                t = time.time()
                tic2 += t - lastTime
                lastTime = t
                # added additional state info below for linear layer (batch)
                x_additional = torch.cat((i_state[2], i_state[3], i_state[4]))
                x_additional = x_additional.reshape(1, -1)
                x_Final_Layer = torch.cat((x, x_additional), dim=-1)
                t = time.time()
                tic3 += t - lastTime
                lastTime = t
                ret_list.append(torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
                t_list.append(torch.relu(self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1))))
                t = time.time()
                tic4 += t - lastTime
            print("tic1:" + str(tic1))
            print("tic2:" + str(tic2))
            print("tic3:" + str(tic3))
            print("tic4:" + str(tic4))
            return ret_list, t_list
        # f_state is a single state
        x = f_state[0]
        x = x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
        x_additional = torch.cat((f_state[2], f_state[3], f_state[4]))
        x_additional = x_additional.reshape(1, -1)  # transpose
        x_Final_Layer = torch.cat((x, x_additional), dim=-1)
        # reminder: state=(img, currentTaskName, maxConsecExecuted, currentlyExecuting, time_left_ep)
        return torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1))), torch.abs(
            self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1)))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# how do these translate to the variables mentioned in the DELFI paper?
# (they dont really this is re-inforcement learning Delfi is normal deep learning)
# (but both use a CNN so some are the same e.g. filter sizes for layers in the CNN)
BATCH_SIZE = 124
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200  # try different values?
TARGET_UPDATE = 10

n_actions = 17

policy_net = DQN(imageWidth, imageHeight, n_actions + 1).to(device)
# outputs should be number of planners +1 which indicates time alloc. to the "best" planner
target_net = DQN(imageWidth, imageHeight, n_actions + 1).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = opt.Adam(policy_net.parameters())
# what optimizer? # switched to ADAM for this project
# optimizer = opt.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = []
    for ns in batch.next_state:
        if ns is not None:
            non_final_next_states.append(ns)

    state_batch = list(batch.state)
    action_batch = list(batch.action)
    reward_batch = []
    for b_rew in batch.reward:
        reward_batch.append(b_rew)
    reward_batch = torch.tensor(reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    # below: get corresponding value from each output vector i in batch corrsponding to action in actionbatch at index i
    # and append them to a list --> state_action_values
    nn_output_vectors, nn_output_time = policy_net(state_batch)
    state_action_values = []
    for i in range(0, len(action_batch)):
        state_action_values.append(nn_output_vectors[i][0][action_batch[i][0].item()])
    state_action_values = torch.stack(state_action_values)  # list of tensors to tensor
    state_time_alloc_Values = torch.stack(nn_output_time)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(len(non_final_next_states), device=device)
    next_state_Values = torch.zeros(BATCH_SIZE, device=device)
    next_planner_time = torch.zeros(BATCH_SIZE, device=device)
    targ_output_planner, targ_output_time = target_net(non_final_next_states)
    for i in range(len(targ_output_planner)):
        next_state_values[i] = torch.max(targ_output_planner[i])

    next_state_Values[non_final_mask] = next_state_values
    next_planner_time[non_final_mask] = torch.tensor(targ_output_time)
    # Compute the expected Q values
    expected_state_action_values = (next_state_Values * GAMMA) + reward_batch  # is this still correct for our example?
    expected_Times = (next_planner_time * GAMMA) + reward_batch  # same question as above??? -(reward_batch)???
    # Compute Huber loss
    criterion = nn.CrossEntropyLoss()
    loss_p = criterion(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
    criterion_t = nn.MSELoss()
    loss_t = criterion_t(state_time_alloc_Values.unsqueeze(1).reshape(124, 1), expected_Times.unsqueeze(1))
    loss = loss_p + loss_t
    # Optimize the model
    optimizer.zero_grad()
    # loss.backward(retain_graph=True)
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def reward(taskIndex, actionNo, actionT, time_left_episode, df):
    # TODO make discussed changes
    minTimeReq = df.iloc[taskIndex][actionNo + 1]  # min time for planner actionNo
    minTimeReq_anyPlanner = df.iloc[taskIndex][1:].min()  # min time out of all planners for given task
    actionTval = actionT.item()  # current time allocated to planner for solving current task
    # actionNo.item+1 because first column is name
    if float(minTimeReq) <= actionTval:
        # interpolate between 0.5 and 1 based on:
        # 0.5*(minTimeBestPlanner/minTimeSelectedPlanner)+0.5
        return torch.tensor([[0.5 * (minTimeReq_anyPlanner / actionT) + 0.5]], dtype=torch.float), True
    elif minTimeReq <= time_left_episode:
        # interpolate betweeen 0 and 0.5 based on:
        #  (1-((t_bestplanner-t_neededcurrentplanner)/t_neededcurrentplanner)) * 0.5
        return torch.tensor([[(1 - ((minTimeReq - minTimeReq_anyPlanner) / minTimeReq)) * 0.5]],
                            dtype=torch.float), False
    elif float(minTimeReq_anyPlanner) <= actionTval:
        return torch.tensor([[-0.5]], dtype=torch.float), False
    else:
        return torch.tensor([[-1]], dtype=torch.float), False


resize = T.Compose([T.ToPILImage(),
                    T.Resize(imageWidth, interpolation=Image.CUBIC),
                    T.ToTensor()])


def select_action(select_action_State):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was found
            planner_vector, best_planner_runtime = policy_net(select_action_State)
            return planner_vector.max(1)[1].view(1, 1), best_planner_runtime  # .detach()
    else:
        timeAlloc = random.random() * select_action_State[4]  # random allocation between 0 and remaining time
        actionNo = torch.tensor([[random.randrange(n_actions)]], device=device)
        return actionNo, torch.tensor([[timeAlloc]])  # .detach()


def randAction(timeLeft):
    timeAlloc = random.random() * timeLeft
    actionNo = torch.tensor([[random.randrange(n_actions)]], device=device)
    return actionNo, torch.tensor([[timeAlloc]])


testSet = p.read_csv(
    "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/testing.csv")


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, randAverageReward):
    print("start testing...")

    minTimeReq_best_planner_list_test = testSet.min(axis=1)
    num_of_tests = len(testSet)
    rewardTotal = 0
    number_of_passes = 0
    number_correct = 0
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        e_time_left_ep = 1800
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
            actionVector, Action_t = policy_net(state)
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
                state = (e_img, e_current_task_index, e_maxConsecExecuted, e_currentlyExecuting, torch.tensor([e_time_left_ep]))
            else:
                number_correct += 1
                # action leads to goal
                break
            prevActionIdx = action_idx
    averageRewards.append((rewardTotal / number_of_passes).item())
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


# testSet = p.read_csv(
#     "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/testing.csv")
#
# print("start testing...")
# tic = time.time()
#
# minTimeReq_best_planner_list_test = testSet.min(axis=1)
# num_of_tests = len(testSet)
# rewardTotal = 0
# rand_rewardTotal = 0
# number_of_passes = 0
# number_correct = 0
# rand_number_correct = 0
# number_incorrect = 0
# rand_number_incorrect = 0
# averageRewards = []
# randAverageRewards = []
# passes = []
# for task_i_idx in range(num_of_tests):
#     time_left_ep = 1800
#     maxConsecExecuted = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
#     currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float)
#     current_task_index = task_i_idx
#     currentTaskName = testSet.iloc[current_task_index][0]
#     currentTaskLoc = taskFolderLoc + currentTaskName + '-bolded-cs.png'
#     img = read_image(currentTaskLoc)
#     img = np.ascontiguousarray(img, dtype=np.float32) / 255
#     img = torch.from_numpy(img)
#     img = resize(img).unsqueeze(0)
#     state = (img, current_task_index, maxConsecExecuted, currentlyExecuting, torch.tensor([time_left_ep]))
#     minTimeReq_best_planner_testSet = minTimeReq_best_planner_list_test[current_task_index]
#     prevActionIdx = None
#     while 0 <= time_left_ep - minTimeReq_best_planner_testSet:
#         actionVector, Action_t = policy_net(state)
#         action_idx = actionVector.max(1)[1].view(1, 1)
#         randAct_idx, randAct_t = randAction(state[4])
#         currReward = reward(current_task_index, action_idx.item(), Action_t, time_left_ep, testSet)[0]
#         rand_currReward = reward(current_task_index, randAct_idx.item(), randAct_t, time_left_ep, testSet)[0]
#         rewardTotal += currReward
#         rand_rewardTotal += rand_currReward
#         averageRewards.append(rewardTotal / number_of_passes)
#         randAverageRewards.append(rand_rewardTotal / number_of_passes)
#         passes.append(number_of_passes)
#         number_of_passes += 1
#         # actionNo.item+1 because first column is name
#         if testSet.iloc[current_task_index][action_idx.item() + 1] > randAct_t:
#             rand_number_incorrect += 1
#         else:
#             rand_number_correct += 1
#         if testSet.iloc[current_task_index][action_idx.item() + 1] > Action_t:
#             number_incorrect += 1
#             # action hasnt led to goal continue
#             if prevActionIdx is action_idx:
#                 currentlyExecuting[action_idx] += Action_t.item()
#             else:
#                 currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                                   dtype=torch.float)
#                 currentlyExecuting[action_idx] += Action_t.item()
#             time_left_ep -= Action_t
#             if currentlyExecuting[action_idx] > maxConsecExecuted[action_idx]:
#                 maxConsecExecuted[action_idx] = currentlyExecuting[action_idx]
#             state = (img, current_task_index, maxConsecExecuted, currentlyExecuting, torch.tensor([time_left_ep]))
#         else:
#             number_correct += 1
#             # action leads to goal
#             break
#         prevActionIdx = action_idx
# print("average reward=" + str((rewardTotal / number_of_passes).item()))
# print("percentage correct=" + str((number_correct / number_of_passes) * 100) + "%")
# print("percentage incorrect=" + str((number_incorrect / number_of_passes) * 100) + "%")
# print("randomAction percentage correct=" + str((rand_number_correct / number_of_passes) * 100) + "%")
# print("randomAction percentage incorrect=" + str((rand_number_incorrect / number_of_passes) * 100) + "%")
# print("number of passes=" + str(number_of_passes))
# # plotting average reward development throughout test set
# plt.plot(passes, averageRewards, label="policy network")
# plt.plot(passes, randAverageRewards, label="random decisions")
# plt.xlabel('number of calls to forward')
# plt.ylabel('average reward')
# plt.title('average rewards while testing DQN:')
# plt.legend()
# plt.show()
#
# print("finished testing in " + str(time.time() - tic) + " seconds")
# print()
# print()

# TRAINING
num_episodes = 3000  # up to 4000 if possible later

minTimeReq_best_planner_list_train = trainingSet.min(axis=1)

episodeList = []
averageRewardList = []
for i_episode in range(num_episodes):
    print("episode " + str(i_episode))
    time_left_ep = 1800
    maxConsecExecuted = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # .detach()
    currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # .detach()
    current_task_index = math.floor(random.random() * len(trainingSet))  # pick a random task in training set
    currentTaskName = trainingSet.iloc[current_task_index][0]
    currentTaskLoc = taskFolderLoc + currentTaskName + '-bolded-cs.png'
    img = read_image(currentTaskLoc)
    # convert image to Tensor below for CNN
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    img = resize(img).unsqueeze(0)
    state = (img, current_task_index, maxConsecExecuted, currentlyExecuting, torch.tensor([time_left_ep]))  # .detach()
    last_actionNumber = None
    same_action = False
    num_passes = 0
    minTimeReq_best_planner_trainSet = minTimeReq_best_planner_list_train[current_task_index]
    while 0 <= time_left_ep - minTimeReq_best_planner_trainSet:
        num_passes += 1
        # Select and perform an action
        action = select_action(state)
        actionNumber = action[0].item()
        actionTime = action[1]
        if last_actionNumber == actionNumber:  # to update consecutive times below
            same_action = True
        rewardVal, done = reward(current_task_index, actionNumber, actionTime, time_left_ep, trainingSet)
        if not done:
            time_left_ep -= actionTime
            if same_action:
                currentlyExecuting[actionNumber] += actionTime
            else:
                currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # .detach()
                currentlyExecuting[actionNumber] = currentlyExecuting[actionNumber] + actionTime
            max_index = max(range(len(currentlyExecuting)), key=currentlyExecuting.__getitem__)
            maxConsecExecutedNext = state[2].clone()
            if maxConsecExecutedNext[max_index] < currentlyExecuting[max_index]:
                maxConsecExecutedNext[max_index] = currentlyExecuting[max_index]
            # create new state
            next_state = (
                img, current_task_index, maxConsecExecutedNext, currentlyExecuting,
                torch.tensor([time_left_ep]))  # .detach()

        else:
            # next state is final
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, rewardVal)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            break
    print(num_passes)
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        print("update target network & Test network...")
        target_net.load_state_dict(policy_net.state_dict())
        episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, -0.25)

print('Completed training...')
