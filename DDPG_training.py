import torch
import numpy as np
import pandas as p

import portfolio_environment
from DDPG_reward import reward
from Replay_Memory_and_utils import ReplayMemory, Transition, resize, imageWidth, imageHeight
from DDPG_evaluation import evaluateNetwork
from DDPG import DDPG

# create args object to hold all parameters
# in evaluation finish task(unsucessful) if same planner repeatedly chosen with bad time?state not ident. continue?
# TODO limit steps per ep

trainingSet = p.read_csv("IPC-image-data-master/problem_splits/training.csv")
taskFolderLoc = "IPC-image-data-master/grounded/"

gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)

BATCH_SIZE = 32
EVALUATE = 10

memory = ReplayMemory(10000)
agent = DDPG(gamma=gamma, tau=tau, h=imageHeight, w=imageWidth)
env = portfolio_environment.PortfolioEnvironment(trainingSet, taskFolderLoc, 1800, reward, 128, 128)

# calculate random action baseline prior to TRAINING
_, average_Reward = evaluateNetwork([], [], 0, agent, 0, rand_bool=True)
rand_a_baseline = average_Reward[0]

time_per_ep = 1800
num_episodes = 3000  # up to 4000 if possible later

episodeList = []
averageRewardList = []

# TRAINING
for i_episode in range(num_episodes):
    print("episode " + str(i_episode))
    # obs is a dict
    obs, _ = env.reset()
    task_img = torch.from_numpy(obs.get('task_img'))
    img = resize(task_img)
    state = img.unsqueeze(0)
    state_additional = torch.cat((torch.tensor(env.max_time_executed, dtype=torch.float32),
                                  torch.tensor(env.consecutive_time_running, dtype=torch.float32),
                                  torch.tensor([env.time_left], dtype=torch.float32)))
    num_passes = 0
    final_state = False
    time_restriction = False
    while not final_state and not time_restriction:
        num_passes += 1
        # Select and perform an action
        actions, actionTime = agent.act(state, state_additional)
        actionNumber = torch.argmax(actions).item()
        # print("actionTime: ", actionTime)
        # print("action number: ", actionNumber)
        env_action = np.concatenate(((np.array(actions.detach())).reshape((17,)), np.array(actionTime.detach())))
        obs, rewardVal, final_state, time_restriction, _ = env.step(env_action)
        next_state = state
        next_state_additional = torch.tensor(obs.get('task_additional'), dtype=torch.float32)
        # Store the transition in memory
        mask = torch.Tensor([final_state])
        rewardVal = torch.tensor(rewardVal, dtype=torch.float32)
        memory.push(state, state_additional, env.task_idx, actions, actionTime, mask, next_state,
                    next_state_additional, rewardVal)

        # Move to the next state
        state = next_state
        state_additional = next_state_additional
        # Perform one step of the optimization (on the policy network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))
            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update(batch)  # optimize network/s

    if i_episode % EVALUATE == 0:
        if len(memory) >= BATCH_SIZE:
            print("testing network...")
            episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, agent,
                                                             rand_a_baseline)

print('Completed training...')
