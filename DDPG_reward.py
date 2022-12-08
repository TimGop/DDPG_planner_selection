import numpy as np


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, plannerCurrTime_unclipped, plannerCurrPrevConsecutiveTime, time_left_episode, df,
           ominicron=10, Theta=10, Epsilon=1, time_per_ep=1800):
    # determine the boundaries
    plannerCurrTime = plannerCurrTime_unclipped if plannerCurrTime_unclipped < time_left_episode else time_left_episode
    minTimeReq_Planners = np.array(df.iloc[taskIndex][1:])
    minTimeReq_Planners[plannerCurrNo] = minTimeReq_Planners[plannerCurrNo] - plannerCurrPrevConsecutiveTime
    min_val = np.min(minTimeReq_Planners)
    lower_bound = min_val * 1.1
    if minTimeReq_Planners[plannerCurrNo] <= time_left_episode:
        if minTimeReq_Planners[plannerCurrNo] == min_val:
            upper_bound = min_val * 1.2
        else:
            upper_bound = minTimeReq_Planners[plannerCurrNo] * 1.2
    else:
        upper_bound = lower_bound + 100  # iff cant solve with current planner still reward good time allocation
    if upper_bound - lower_bound <= 50:
        upper_bound = lower_bound + 50
    if upper_bound > time_left_episode:
        upper_bound = time_left_episode-0.1
        # upper bound cant be exactly t_left or overestimating will give good reward

    # calculate the reward
    if plannerCurrTime <= 0:
        R = -10
    elif min_val < plannerCurrTime <= lower_bound:
        R = ((plannerCurrTime - min_val)/(lower_bound - min_val))
    elif lower_bound < plannerCurrTime <= upper_bound:
        R = 1
    elif upper_bound < plannerCurrTime < time_left_episode:
        R = (plannerCurrTime-time_left_episode)/(upper_bound-time_left_episode)
    elif plannerCurrTime == time_left_episode:
        R = 0
    else:
        R = 0

    # check if done
    done = minTimeReq_Planners[plannerCurrNo] <= plannerCurrTime <= time_left_episode

    return np.array([R]), done


class Reward:
    pass
