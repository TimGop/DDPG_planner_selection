import torch
import numpy as np

ominicron = 1
Theta = 1
Epsilon = 0.2
time_per_ep = 1800  # TODO have this passed


def reward(taskIndex, actionNo, actionT, time_left_episode, df):
    if actionT > time_left_episode:
        return torch.tensor([-Theta - Epsilon], dtype=torch.float), False
    if actionT <= 0:
        return torch.tensor([-ominicron], dtype=torch.float), False  # action time is zero
    # print("action number: " + str(actionNo))
    minTimeReq_currPlanner = df.iloc[taskIndex][actionNo + 1]
    minTimeReq_anyPlanner = df.iloc[taskIndex][1:].min()

    if minTimeReq_currPlanner > time_left_episode:
        # bad planner
        if (time_left_episode - actionT) > minTimeReq_anyPlanner:
            # time left for another planner --> ok bad
            return torch.tensor([-Epsilon], dtype=torch.float), False
        else:
            # no time left for other planner --> very bad
            return torch.tensor([-Theta - Epsilon], dtype=torch.float), False

    else:
        R_p = (np.min([actionT, minTimeReq_currPlanner]) / minTimeReq_currPlanner) * Epsilon
        R_s = 0
        # good planner
        if actionT >= minTimeReq_currPlanner:
            # solved --> very good
            R_s = (1 - (actionT - minTimeReq_currPlanner) / (time_per_ep - minTimeReq_currPlanner)) * Theta
        # else R_s stays equal to zero
        return torch.tensor([R_s + R_p], dtype=torch.float), True


class Reward:
    pass
