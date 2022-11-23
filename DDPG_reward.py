import numpy as np

ominicron = 10
Theta = 10
Epsilon = 1
time_per_ep = 1800  # TODO have this passed


# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, df):
    # We can calculate the reward however we like, but I do not care about selecting
    # the best in the end, but for the final evaluation about solving!
    runtimes = df.iloc[taskIndex][:17]
    best_time = runtimes.min()
    curr_time = runtimes[plannerCurrNo]
    if best_time > 1800:  # actually, this should not happen, but currently it does
        return 0, True
    assert best_time < 1800, best_time

    unnecessary_time = curr_time - best_time
    buffer_time = 1800 - best_time

    if curr_time >= 1800:
        return - Theta - Epsilon, False
    else:
        return Epsilon + (1 - (unnecessary_time / buffer_time)) * Theta, True


class Reward:
    pass
