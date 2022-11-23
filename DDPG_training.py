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

trainingSet = p.read_csv("IPC-image-data-master/problem_splits/testing.csv")
taskFolderLoc = "IPC-image-data-master/lifted/"

gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)

BATCH_SIZE = 16
EVALUATE = 10
memory = ReplayMemory(10000)
env = portfolio_environment.PortfolioEnvironment(trainingSet, taskFolderLoc, reward)
agent = DDPG(gamma=gamma, tau=tau, h=imageHeight, w=imageWidth, env=env)

# calculate random action baseline prior to TRAINING
_, average_Reward = evaluateNetwork([], [], 0, agent, 0, rand_bool=True)
rand_a_baseline = average_Reward[0]

time_per_ep = 1800
num_episodes = 3000  # up to 4000 if possible later

episodeList = []
averageRewardList = []

# TRAINING
for i_episode in range(num_episodes):
    print("\nepisode " + str(i_episode))
    # obs is a dict
    obs, _ = env.reset()
    task_img = torch.from_numpy(obs)
    img = resize(task_img)
    state = img.unsqueeze(0)
    final_state = False
    # limit to 5 attempts per episode
    for i in range(1):
        # print("i", i)
        # Select and perform an action
        actions = agent.act(state)
        # assert False, actions
        actionNumber = torch.argmax(actions).item()
        print(actionNumber, end=",")
        # print("actionTime: ", actionTime)
        # print("action number: ", actionNumber)
        env_action = (np.array(actions.detach())).reshape((17,))
        next_state, rewardVal, final_state, episode_end, _ = env.step(env_action)

        mask = torch.Tensor([episode_end])
        rewardVal = torch.tensor(rewardVal, dtype=torch.float32)

        # Store the transition in memory
        memory.push(state, env.task_idx, actions, mask, state, rewardVal)



        # Perform one step of the optimization (on the policy network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))
            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update(batch)  # optimize network/s
        if episode_end:
            break
    if i_episode % EVALUATE == 0:
        if len(memory) >= BATCH_SIZE:
            print("testing network...")
            episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, agent,
                                                             rand_a_baseline)

print('Completed training...')
