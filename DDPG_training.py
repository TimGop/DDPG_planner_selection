import math
import torch
import numpy as np
import pandas as p
import random
from DDPG_reward import reward
from Replay_Memory_and_utils import ReplayMemory, Transition, resize, imageWidth, imageHeight
from DDPG_evaluation import evaluateNetwork

from torchvision.io import read_image
from DDPG import DDPG

# change actor network back to layers used and output from the DQN Q network DONE?
# reward=-omnicron t=0 and epsiode termination DONE?
# implement training action method (very similiar to the action method in DQN_old_code.py) DONE?
# finish evaluaton method DONE?
# TODO fix compatibility errors in agent.update()
# TODO in evaluation finish task(unsucessful) if same planner repeatedly chosen with bad time?
# TODO create args object to hold all parameters
# TODO BONUS: create custom enviroment with openAI gym

trainingSet = p.read_csv(
    "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/training.csv")
taskFolderLoc = "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/grounded/"

gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)

BATCH_SIZE = 124
GAMMA = 0.8
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200  # try different values?
EVALUATE = 10

memory = ReplayMemory(10000)

agent = DDPG(gamma=gamma, tau=tau, h=imageHeight, w=imageWidth)

# calculate random action baseline prior to TRAINING
_, average_Reward = evaluateNetwork([], [], 0, agent, 0, rand_bool=True)
rand_a_baseline = average_Reward[0]

# TRAINING

time_per_ep = 1800
num_episodes = 3000  # up to 4000 if possible later

minTimeReq_best_planner_list_train = trainingSet.min(axis=1)

episodeList = []
averageRewardList = []

for i_episode in range(num_episodes):
    print("episode " + str(i_episode))
    time_left_ep = time_per_ep
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
    state = img
    state_additional = torch.cat((maxConsecExecuted, currentlyExecuting,
                                  torch.tensor([time_left_ep])))
    last_actionNumber = None
    same_action = False
    num_passes = 0
    minTimeReq_best_planner_trainSet = minTimeReq_best_planner_list_train[current_task_index]
    while 0 <= time_left_ep - minTimeReq_best_planner_trainSet:
        num_passes += 1
        # Select and perform an action
        action = agent.act(state, state_additional)
        actionNumber = round(action[0].item())  # conversion to int
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
            maxConsecExecutedNext = maxConsecExecuted
            if maxConsecExecutedNext[max_index].item() < currentlyExecuting[max_index].item():
                maxConsecExecutedNext[max_index] = currentlyExecuting[max_index]
            # create new state
            next_state = img
            # next_state_additional = (maxConsecExecutedNext, currentlyExecuting,
            #                          torch.tensor([time_left_ep]))
            next_state_additional = torch.cat((maxConsecExecutedNext, currentlyExecuting,
                                               torch.tensor([time_left_ep])))
            maxConsecExecuted = maxConsecExecutedNext
        else:
            # next state is final
            next_state = None
            next_state_additional = None

        # Store the transition in memory
        mask = torch.Tensor([done])
        memory.push(state, state_additional, current_task_index, action, mask, next_state, next_state_additional,
                    rewardVal)

        # Move to the next state
        state = next_state
        state_additional = next_state_additional
        # Perform one step of the optimization (on the policy network)
        if len(memory) > BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))

            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update(batch)  # optimize network/s

        if done or actionTime == 0:
            break
    if i_episode % EVALUATE == 0:
        print("testing network...")
        if len(memory) >= BATCH_SIZE:
            print()
            episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, agent,
                                                             rand_a_baseline)

print('Completed training...')
