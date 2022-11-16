import torch

ominicron = 10
Theta = 10
Epsilon = 1
time_per_ep = 1800  # TODO have this passed


def reward(taskIndex, plannerCurrNo, plannerCurrTime, plannerCurrPrevConsecutiveTime, time_left_episode, df):
    R_t = 0
    # Patrick: plannerCurrPrevConsecutiveTime is the consecutive time, excluding plannerCurrTime,
    # of the now selected planner
    if plannerCurrTime <= 0:
        return torch.tensor([-ominicron], dtype=torch.float), False  # action time is zero
    if plannerCurrTime > time_left_episode:  # Patrick: Limited predictionto episode
        # Patrick: Let's keep it simple and not penalaize this yet.  --> penalized now
        R_t = -5
        plannerCurrTime = time_left_episode
        # return torch.tensor([-Theta - Epsilon], dtype=torch.float), False
    plannerCurrConsecutiveTime = plannerCurrPrevConsecutiveTime + plannerCurrTime

    # print("action number: " + str(plannerCurrNo))
    minTimeReq_currPlanner = df.iloc[taskIndex][plannerCurrNo + 1]
    minTimeReq_anyPlanner = df.iloc[taskIndex][1:].min()
    missingTime = minTimeReq_currPlanner - plannerCurrConsecutiveTime  # negative means solved!

    # Patrick: Previous check ignored consecutive previous runtime!
    # Choosen planner cannot solve within remaining time
    if missingTime > time_left_episode:
        # bad planner
        if (time_left_episode - plannerCurrTime) > minTimeReq_anyPlanner:
            # time left for another planner --> ok bad
            return torch.tensor([-Epsilon + R_t], dtype=torch.float), False
        else:
            # no time left for other planner --> very bad
            return torch.tensor([-Theta - Epsilon + R_t], dtype=torch.float), False

    else:
        # Patrick: Let's keep it simple
        # R_p = (np.min([plannerCurrTime, minTimeReq_currPlanner]) / minTimeReq_currPlanner) * Epsilon
        if missingTime <= 0:
            R_p = Epsilon
            # Patrick: Positive reward also depends on previous bad choices!
            passed_time = time_per_ep - time_left_episode
            unnecessary_passed_time = passed_time - minTimeReq_anyPlanner
            time_buffer = time_per_ep - minTimeReq_anyPlanner
            R_s = (time_buffer - unnecessary_passed_time) / time_buffer * Theta
            # Patrick: the reward should speak about the best possible planner R_s = (1 - (plannerCurrConsecutiveTime
            # - minTimeReq_currPlanner) / (time_per_ep - minTimeReq_currPlanner)) * Theta
            return torch.tensor([R_s + R_p + R_t], dtype=torch.float), True
        else:
            R_p = 0
            R_s = 0
            return torch.tensor([R_s + R_p + R_t], dtype=torch.float), False
        # else R_s stays equal to zero


class Reward:
    pass
