# DQN_planner_selection
train a DQN to select the best planner for a given planning task
Tasks are encoded as images in a grounded or lifted representation (128x128x1 PNG i.e. 128x128 greyscale PNG)
potential possibility of using a GNN(with graphs) in the future instead of a CNN(with images) because images lose information (add accidental complexity while removing some essential complexity) 
whereas a graph with a GNN does not share this problem.


# Now using a DDPG instead of a DQN
To run DDPG run the file DDPG_training.py
