import torch
import numpy as np
import pandas as p
import argparse
import portfolio_environment
import time
from DDPG_reward import reward
from Replay_Memory_and_utils import ReplayMemory, Transition, resizer
from MARL_evaluation import evaluation
from DDPG import DDPG
from DQN import DQN

trainingSet = p.read_csv("IPC-image-data-master/problem_splits/training.csv")
taskFolderLoc = "IPC-image-data-master/grounded/"

parser = argparse.ArgumentParser()
parser.add_argument("--BATCH_SIZE", default=32, type=int,
                    help="Size of the batch used in training to update the networks(default: 32)")
parser.add_argument("--num_episodes", default=3000, type=int,
                    help="Num. of total timesteps of training (default: 3000)")
parser.add_argument("--gamma", default=0.99,
                    help="Discount factor (default: 0.99)")
parser.add_argument("--tau", default=0.001,
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
parser.add_argument("--width", default=128, type=int,
                    help="Image width used(default: 128)")
parser.add_argument("--height", default=128, type=int,
                    help="Image height used(default: 128)")
parser.add_argument("--mem_size", default=10000, type=int,
                    help="Size of the Replay Memory(default: 10000)")
parser.add_argument("--p_SD", default=0.7, type=float,
                    help="Planner noise Standard deviation(default: 0.7)")
parser.add_argument("--t_SD", default=200, type=float,
                    help="Time noise Standard deviation(default: 200)")
parser.add_argument("--timeout", default=time.time() + 60 * 60 * 24,
                    help="time limit(default: 10 min)")
parser.add_argument("--eps_start", default=0.9,
                    help="used for random actions in DQN")
parser.add_argument("--eps_end", default=0.9,
                    help="used for random actions in DQN")
parser.add_argument("--eps_decay", default=0.9,
                    help="used for random actions in DQN")
parser.add_argument("--target_update", default=10, type=int,
                    help="number of iterations before dqn_target is receives hard update")
args = parser.parse_args()

memory = ReplayMemory(args.mem_size)
env = portfolio_environment.PortfolioEnvironment(trainingSet, taskFolderLoc, args.time_per_ep, reward, t_SD=args.t_SD,
                                                 p_SD=args.p_SD, nb_planners=args.num_planners, omnicron=args.omnicron,
                                                 Theta=args.Theta, Epsilon=args.Epsilon, time_per_ep=args.time_per_ep)
agent_dqn = DQN(args.BATCH_SIZE, args.width, args.height, args.gamma, args.eps_start, args.eps_end, args.eps_decay,
                args.num_planners)
agent_ddpg = DDPG(gamma=args.gamma, tau=args.tau, h=args.height, w=args.width, env=env, num_planners=args.num_planners)
resizer = resizer(args.width)  # args.width should equal args.height
# calculate random action baseline prior to TRAINING
evaluater = evaluation(args.time_per_ep, t_SD=args.t_SD, p_SD=args.p_SD, w_and_h=args.width,
                       nb_planners=args.num_planners, omnicron=args.omnicron, Theta=args.Theta, Epsilon=args.Epsilon)

_, average_Reward = evaluater.evaluateNetwork(episodeNumbers=[], averageRewards=[], currentEpisodeNumber=0,
                                              discrete_agent=agent_dqn, randAverageReward=0, rand_bool=True,
                                              n_actions=args.num_planners, continuous_agent=agent_ddpg)
rand_a_baseline = average_Reward[0]

episodeList = []
averageRewardList = []
i_episode = 0
max_average_reward = -10 ** 10  # start with very very negative(assume first reward already better)

# TRAINING
while time.time() < args.timeout:
    print("episode " + str(i_episode))
    # obs is a dict
    obs, _ = env.reset()
    task_img = torch.from_numpy(obs.get('task_img'))
    img = resizer.resize(task_img)
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
        action = agent_dqn.select_action(state, state_additional, num_passes)
        actionTime = agent_ddpg.act(state, state_additional, action)
        print("actionTime: ", actionTime)
        print("action number: ", action)
        env_action = np.concatenate(
            ((np.array(action.detach().squeeze(0))), np.array(actionTime.detach())))
        obs, reward, final_state, time_restriction, _ = env.step(env_action)
        reward_planner, reward_time = reward
        print(final_state)
        print(time_restriction)
        next_state = state
        next_state_additional = torch.tensor(obs.get('task_additional'), dtype=torch.float32)
        mask = torch.Tensor([final_state])
        reward_planner = torch.tensor(reward_planner, dtype=torch.float32)
        reward_time = torch.tensor(reward_time, dtype=torch.float32)

        # Store the transition in memory
        memory.push(state, state_additional, env.task_idx, action, actionTime, mask, next_state,
                    next_state_additional, reward_planner, reward_time)

        # Move to the next state
        state = next_state
        state_additional = next_state_additional
        # Perform one step of the optimization (on the policy network)
        # TODO update DQN here
        if len(memory) >= args.BATCH_SIZE:
            transitions = memory.sample(args.BATCH_SIZE)
            # Transpose the batch
            # (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
            batch = Transition(*zip(*transitions))
            # Update actor and critic according to the batch
            value_loss, policy_loss = agent_ddpg.update(batch)  # optimize network/s

    if i_episode % args.EVALUATE == 0:
        if len(memory) >= args.BATCH_SIZE:
            print("testing network...")
            episodeList, averageRewardList = evaluater.evaluateNetwork(episodeList, averageRewardList, i_episode,
                                                                       agent_dqn, agent_ddpg, rand_a_baseline)
            if max_average_reward < averageRewardList[-1]:
                max_average_reward = averageRewardList[-1]
                torch.save(agent_dqn.policy_net.state_dict(), "net_configs/dqn.pth")
                torch.save(agent_ddpg.actor.state_dict(), "net_configs/actor.pth")
                torch.save((agent_ddpg.critic.state_dict()), "net_configs/critic.pth")
    i_episode += 1

print('Completed training...')
