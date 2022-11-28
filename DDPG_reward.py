# used in openAI gym enviroment where np arrays are used
def reward(taskIndex, plannerCurrNo, df, time_per_ep, ominicron, Theta, Epsilon):
    # We can calculate the reward however we like, but I do not care about selecting
    # the best in the end, but for the final evaluation about solving!
    runtimes = df.iloc[taskIndex]
    best_time = runtimes.min()
    curr_time = runtimes[plannerCurrNo]
    assert best_time < time_per_ep, "ERROR: unsolvable task found"
    unnecessary_time = curr_time - best_time
    buffer_time = time_per_ep - best_time

    if curr_time >= time_per_ep:
        return - Theta - Epsilon, False
    else:
        return Epsilon + (1 - (unnecessary_time / buffer_time)) * Theta, True


class Reward:
    pass
