import numpy as np

ominicron = 10
Theta = 10
Epsilon = 1
time_per_ep = 1800  # TODO have this passed


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, df):
    is_best = False
    best_to_worst_planner_indices = np.argsort(df.iloc[taskIndex][1:])
    current_pos_in_list = np.where(best_to_worst_planner_indices == plannerCurrNo)[0]
    if current_pos_in_list == 0:
        is_best = True
    rew = ((17 - current_pos_in_list) * 0.1) - 0.9

    return rew, is_best


class Reward:
    pass
