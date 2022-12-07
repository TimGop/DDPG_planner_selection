import numpy as np


def reward(taskIndex, plannerCurrNo, plannerCurrPrevConsecutiveTime, time_left_episode, df,
           ominicron=10, Theta=10, Epsilon=1):
    minTimeReq_Planners = df.iloc[taskIndex][1:]
    minTimeReq_Planners[plannerCurrNo] = minTimeReq_Planners[plannerCurrNo] - plannerCurrPrevConsecutiveTime
    planner_times_sorted = sorted(minTimeReq_Planners)
    planner_indices_sorted = sorted(range(len(minTimeReq_Planners)), key=lambda k: minTimeReq_Planners[k])
    max_pos_still_solved = last_index_solved(planner_times_sorted, time_left_episode)
    position_current_planner_in_sorted = planner_indices_sorted.index(plannerCurrNo)
    if max_pos_still_solved >= position_current_planner_in_sorted:
        if max_pos_still_solved == position_current_planner_in_sorted and max_pos_still_solved == 0:
            rew = 1  # to avoid special case of divide by zero
        else:
            rew = ((max_pos_still_solved - position_current_planner_in_sorted) / max_pos_still_solved) * 0.5 + 0.5
        done = True
    else:
        rew = 0
        done = False
    return np.array([rew]), done


def last_index_solved(arr, time_left_ep):
    next_val = next((x for x in arr if x > time_left_ep), None)
    return 16 if next_val is None else arr.index(next_val) - 1
