import numpy as np


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, plannerCurrTime, plannerCurrPrevConsecutiveTime, time_left_episode, df,
           ominicron=10, Theta=10, Epsilon=1, time_per_ep=1800):
    if plannerCurrTime > time_left_episode:
        plannerCurrTime = time_left_episode
    if plannerCurrTime <= 0:
        return -ominicron, False

    minTimeReq_currPlanner = df.iloc[taskIndex][plannerCurrNo + 1]
    if minTimeReq_currPlanner <= plannerCurrTime + plannerCurrPrevConsecutiveTime:
        return Theta, True
    else:
        return -Epsilon, False


class Reward:
    pass
