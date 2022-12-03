# planner portfolios using deep multi-agent RL:
To run the code run the file training.py.
Offline planning portfolios can decide on a planner and time allocation given a task. However this is done with no updated knowledge about the state such as planners previously attempted. Here we explore an online learning of planning portfolios making use of recent advances in deep RL that extend DQN to continuos action spaces (i.e. DDPG or deep deterministic policy gradients).
In this versio of the project we use make use of MARL (i.e. deep multi-agent reinforcement learning) by creating seperate DQN and DDPG agents that estimate seperate components of the complete action.
