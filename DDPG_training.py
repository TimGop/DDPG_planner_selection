import torch
import line_environment
from Replay_Memory_and_utils import ReplayMemory, Transition
from DDPG_evaluation import evaluateNetwork
from DDPG import DDPG


gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)

BATCH_SIZE = 128
EVALUATE = 30

memory = ReplayMemory(10000)
env = line_environment.lineEnvironment()
agent = DDPG(gamma=gamma, tau=tau, env=env)

num_episodes = 3000  # up to 4000 if possible later

episodeList = []
averageRewardList = []

# TRAINING
for i_episode in range(num_episodes):
    print("episode " + str(i_episode))
    state, _ = env.reset()
    # print("initial_state:", state)
    state = torch.tensor([state], dtype=torch.float32)
    num_passes = 0
    while True:
        num_passes += 1
        # Select and perform an action
        action = agent.act(state)
        next_state, rewardVal, _, game_end, _ = env.step(action.item())
        next_state = torch.tensor([next_state], dtype=torch.float32)
        # Store the transition in memory
        mask = torch.Tensor([game_end])
        rewardVal = torch.tensor([rewardVal], dtype=torch.float32)
        memory.push(state, action, mask, next_state, rewardVal)
        # Move to the next state
        state = next_state
        # print("action", action)
        # print("state", state)
        # Perform one step of the optimization (on the policy network)
        if len(memory) >= BATCH_SIZE:
            transitions = memory.sample(BATCH_SIZE)
            batch = Transition(*zip(*transitions))
            # Update actor and critic according to the batch
            value_loss, policy_loss = agent.update(batch)  # optimize network/s
        if game_end:
            break
    if i_episode % EVALUATE == 0:
        if len(memory) >= BATCH_SIZE:
            print("testing network...")
            episodeList, averageRewardList = evaluateNetwork(episodeList, averageRewardList, i_episode, agent)

print('Completed training...')
