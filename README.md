# Now using a DDPG instead of a DQN
To run DDPG run the file DDPG_training.py.
Offline planning portfolios can decide on a planner and time allocation given a task. However this is done with no updated knowledge about the state such as planners previously attempted. Here we explore an online learning of planning portfolios making use of recent advances in deep RL that extend DQN to continuos action spaces (i.e. DDPG or deep deterministic policy gradients).

Note: here the action output of the actor has been extended to produce a time output for each action instead of just one so that the reward could be modified to account for all actions (hopefully more sample efficient learning)
