import math

import numpy as np


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, plannerCurrTime_unclipped, plannerCurrPrevConsecutiveTime, time_left_episode, df,
           ominicron=10, Theta=10, Epsilon=1, time_per_ep=1800):
    # determine the boundaries
    plannerCurrTime = plannerCurrTime_unclipped if plannerCurrTime_unclipped < time_left_episode else time_left_episode
    minTimeReq_Planners = np.array(df.iloc[taskIndex][1:])
    minTimeReq_Planners[plannerCurrNo] = minTimeReq_Planners[plannerCurrNo] - plannerCurrPrevConsecutiveTime
    planner_times_sorted = sorted(minTimeReq_Planners)
    planner_indices_sorted = sorted(range(len(minTimeReq_Planners)), key=lambda k: minTimeReq_Planners[k])
    max_pos_still_solved = last_index_solved(planner_times_sorted, time_left_episode)
    if (max_pos_still_solved + 1) % 2 == 0:
        median = (planner_times_sorted[math.floor(max_pos_still_solved / 2)] + planner_times_sorted[
            math.floor((max_pos_still_solved / 2) + 1)]) / 2
    else:
        median = planner_times_sorted[math.floor(max_pos_still_solved / 2)]
    R = -((plannerCurrTime - median) ** 2)

    # check if done
    done = minTimeReq_Planners[plannerCurrNo] <= plannerCurrTime <= time_left_episode

    return np.array([R]), done


def last_index_solved(arr, time_left_ep):
    next_val = next((x for x in arr if x > time_left_ep), None)
    return 16 if next_val is None else arr.index(next_val) - 1
