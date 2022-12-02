import numpy as np


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, planners, plannerTimes, plannerConsecutiveTimes, time_left_episode, df, n_steps,
           ominicron=10, Theta=10, Epsilon=1, time_per_ep=1800, step_penalty=10):
    current_planner = np.argmax(planners)
    min_times = df.iloc[taskIndex][1:18]
    current_planner_time = plannerTimes[current_planner] if plannerTimes[current_planner] <= time_left_episode \
        else time_left_episode
    done = min_times[current_planner] <= current_planner_time + plannerConsecutiveTimes[
        current_planner]

    R_choice_and_time_and_penalty = 0
    for i in range(len(planners)):
        solvable = min_times[i] <= time_left_episode + plannerConsecutiveTimes[i]
        solves = min_times[i] <= plannerTimes[i] + plannerConsecutiveTimes[i]
        # Reward for predicting correctly if a planner solves a task
        C_i = ominicron if solvable else -(ominicron / 2)
        R_a_i = C_i * (2 * planners[i] - 1)
        # Reward for a good time prediction
        K_i = Theta if solves else -Theta
        R_t_i = planners[i] * K_i

        t_penalty_i = 0  # else t_penalty_i stays equal to 0
        if plannerTimes[i] > time_left_episode:
            t_penalty_i = (-3 * Epsilon)
        elif plannerTimes[i] < 0:
            t_penalty_i = (-5 * Epsilon)
        R_choice_and_time_and_penalty += R_a_i + R_t_i + t_penalty_i
    t_best = max(time_left_episode if solves else current_planner_time, 0)
    R_step_penality = step_penalty  # deinscentivizes too many steps
    R = R_choice_and_time_and_penalty * (t_best/time_per_ep) + R_step_penality
    return R, done


class Reward:
    pass
