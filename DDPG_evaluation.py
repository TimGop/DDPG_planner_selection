import torch

import line_environment as lenv
import matplotlib.pyplot as plt

env = lenv.lineEnvironment()
num_of_tests = 100


def evaluateNetwork(episodeNumbers, averageRewards, currentEpisodeNumber, agent):
    print("start testing...")
    agent.set_eval()
    rewardTotal = 0
    number_of_passes = 0
    episodeNumbers.append(currentEpisodeNumber)

    for task_i_idx in range(num_of_tests):
        state, _ = env.reset()
        state = torch.tensor([state], dtype=torch.float32)
        final_state = False
        pass_per_test = 0
        while not final_state and pass_per_test <= 100:
            number_of_passes += 1
            action = agent.get_action(state)
            next_state, rewardVal, _, final_state, _ = env.step(action)
            next_state = torch.tensor([next_state], dtype=torch.float32)
            state = next_state
            rewardTotal += rewardVal
            pass_per_test += 1

    averageRewards.append((rewardTotal / number_of_passes))
    print("average number of passes per task=" + str(number_of_passes / num_of_tests))
    print(averageRewards)
    if episodeNumbers.__len__() > 1:
        plt.plot(episodeNumbers, averageRewards, color='g', label="policy network")
    else:
        plt.axhline(y=averageRewards[0], color='g', label="policy network")
    plt.xlabel('number of episodes')
    plt.ylabel('average reward')
    plt.title('average rewards while testing DQN:')
    plt.legend()
    plt.show()
    agent.set_train()
    print("finish testing...")
    return episodeNumbers, averageRewards
