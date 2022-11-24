import torch
import random
import numpy as np
import pandas as p

import portfolio_environment
from DDPG_reward import reward
import matplotlib.pyplot as plt
from Replay_Memory_and_utils import resizer


class evaluation:
    def __init__(self, time_per_ep, t_SD, p_SD, w_and_h, nb_planners, omnicron, Theta, Epsilon):
        self.testSet = p.read_csv("IPC-image-data-master/problem_splits/testing.csv")
        self.taskFolderLoc = "IPC-image-data-master/grounded/"
        self.env = portfolio_environment.PortfolioEnvironment(self.testSet, self.taskFolderLoc, time_per_ep, reward,
                                                              t_SD=t_SD, p_SD=p_SD, nb_planners=nb_planners,
                                                              omnicron=omnicron, Theta=Theta, Epsilon=Epsilon,
                                                              time_per_ep=time_per_ep)
        self.resizer = resizer(w_and_h)

    @staticmethod
    def randAction(timeLeft, n_actions):
        return torch.tensor([[random.random() for _ in range(n_actions)]], dtype=torch.float32).softmax(dim=1), \
               torch.tensor([random.random() * timeLeft], dtype=torch.float32)

    def evaluateNetwork(self, episodeNumbers, averageRewards, currentEpisodeNumber, agent, randAverageReward,
                        rand_bool=False,
                        n_actions=17):
        print("start testing...")
        agent.set_eval()
        num_of_tests = len(self.testSet)
        rewardTotal = 0
        number_of_passes = 0
        number_correct = 0
        episodeNumbers.append(currentEpisodeNumber)

        for task_i_idx in range(num_of_tests):
            obs, _ = self.env.reset_testMode(task_i_idx)
            task_img = torch.from_numpy(obs.get('task_img'))
            state = self.resizer.resize(task_img).unsqueeze(0)
            state_additional = torch.cat((torch.tensor(self.env.max_time_executed, dtype=torch.float32),
                                          torch.tensor(self.env.consecutive_time_running, dtype=torch.float32),
                                          torch.tensor([self.env.time_left], dtype=torch.float32)))
            final_state = False
            time_restriction = False
            while not final_state and not time_restriction:
                if not rand_bool:
                    action, action_t = agent.get_action(state, state_additional)
                else:
                    action, action_t = self.randAction(self.env.time_left, n_actions)
                complete_action = np.concatenate((np.array(action.detach().squeeze(0)), np.array(action_t.detach())))
                obs, rewardVal, final_state, time_restriction, _ = self.env.step(complete_action)
                number_of_passes += 1
                if final_state:
                    rewardTotal += rewardVal
                    number_correct += 1
                elif time_restriction:
                    rewardTotal += rewardVal

        averageRewards.append((rewardTotal / num_of_tests))
        print("percentage correct=" + str((number_correct / num_of_tests) * 100) + "%")
        print("average number of passes per task=" + str(number_of_passes / num_of_tests))
        if not rand_bool:
            print(episodeNumbers)
            print(averageRewards)
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
