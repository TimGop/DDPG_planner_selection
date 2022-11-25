import argparse
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

trainingSet = p.read_csv("IPC-image-data-master/problem_splits/training.csv")
taskFolderLoc = "IPC-image-data-master/lifted/"

parser = argparse.ArgumentParser()
parser.add_argument("--BATCH_SIZE", default=32, type=int,
                    help="Size of the batch used in training to update the networks(default: 32)")
parser.add_argument("--num_episodes", default=3000, type=int,
                    help="Num. of total timesteps of training (default: 3000)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.05,
                    help="Update factor for the soft update of the target networks (default: 0.001)")
parser.add_argument("--EVALUATE", default=10, type=int,
                    help="Number of episodes between testing cycles(default: 10)")
parser.add_argument("--time_per_ep", default=1800, type=int,
                    help="The amount of time per episode(default: 1800)")
parser.add_argument("--num_planners", default=17, type=int,
                    help="The number of different planner algorithms(default: 17)")
parser.add_argument("--omnicron", default=10, type=int,
                    help="A constant used for reward calculation(default: 10)")
parser.add_argument("--Theta", default=10, type=int,
                    help="A constant used for reward calculation(default: 10)")
parser.add_argument("--Epsilon", default=1, type=int,
                    help="A constant used for reward calculation(default: 1)")
args = parser.parse_args()

# Initialize memory, environment and agent
memory = ReplayMemory(10000)
env = portfolio_environment.PortfolioEnvironment(trainingSet, taskFolderLoc, reward, time_per_ep=args.time_per_ep,
                                                 omnicron=args.omnicron, Theta=args.Theta, Epsilon=args.Epsilon)
agent = DDPG(gamma=args.gamma, tau=args.tau, h=imageHeight, w=imageWidth, num_planners=args.num_planners, env=env)

# calculate random action baseline prior to TRAINING
_, average_Reward = evaluateNetwork([], [], 0, agent, None, rand_bool=True, n_actions=args.num_planners)
rand_a_baseline = average_Reward[0]

episodeList = []
averageRewardList = []

# TRAINING
for i_episode in range(args.num_episodes):
    print("\nepisode " + str(i_episode))
    # obs is a dict
    obs, _ = env.reset()
    # task_img = torch.from_numpy(obs)
    # img = resize(task_img)
    state = torch.tensor([obs], dtype=torch.float32)  # img.unsqueeze(0)
    final_state = False
    # limit to 1 attempt per episode
    for i in range(1):
        # Select and perform an action
        actions = agent.act(state)
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
        if len(memory) >= args.BATCH_SIZE:
            transitions = memory.sample(args.BATCH_SIZE)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))
            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update(batch)  # optimize network/s
        if episode_end:
            break
    if i_episode % args.EVALUATE == 0:
        if len(memory) >= args.BATCH_SIZE:
            print("testing network...")
            episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, agent,
                                                             rand_a_baseline)

print('Completed training...')
