import math

import numpy as np
import pandas as p
import torch
import torch.nn as nn
import torch.optim as opt
from itertools import count
import random
from collections import namedtuple, deque
from torchvision.io import read_image
import torchvision.transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
State = namedtuple('State', ('img', 'currentTaskIndex', 'maxConsecExecuted', 'currentlyExecuting', 'time_left_ep'))

trainingSet = p.read_csv(
    "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/problem_splits/training.csv")


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv2d = nn.Conv2d(1, 128, kernel_size=(2, 2), stride=(1, 1))
        self.maxPool = nn.MaxPool2d(kernel_size=1)
        self.flatten = nn.Flatten()
        self.dropOut = nn.Dropout(p=0.49)
        num = 2064547 - h - w
        linear_input_size = num + h + w  # essentially just used this to remove unused args
        self.headPlanner = nn.Linear(linear_input_size, outputs - 1)
        self.headTime = nn.Linear(linear_input_size, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x_image, x_task_index, x_ConsecExecuted, x_currentlyExecuting, x_time_left_ep):
        x = x_image
        x = x.to(device)
        x = self.dropOut(self.flatten(self.maxPool(self.conv2d(x))))
        if len(x_time_left_ep) > 50:  # is a batch
            x_additional = torch.cat((x_ConsecExecuted, x_currentlyExecuting, x_time_left_ep), dim=1)
        else:
            x_additional = torch.cat((x_ConsecExecuted, x_currentlyExecuting, x_time_left_ep), dim=0)
            x_additional = x_additional.reshape(1, -1)  # transpose
        x_Final_Layer = torch.cat((x, x_additional), dim=-1)
        # reminder: state=(img, currentTaskIndex, maxConsecExecuted, currentlyExecuting, time_left_ep)
        outp = torch.sigmoid(self.headPlanner(x_Final_Layer.view(x_Final_Layer.size(0), -1)))
        outt = torch.relu(self.headTime(x_Final_Layer.view(x_Final_Layer.size(0), -1)))
        return outp, outt


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# how do these translate to the variables mentioned in the DELFI paper?
# (they dont really this is re-inforcement learning Delfi is normal deep learning)
# (but both use a CNN so some are the same e.g. filter sizes for layers in the CNN)
BATCH_SIZE = 124
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = 17

policy_net = DQN(128, 128, n_actions + 1).to(device)
# outputs should be number of planners +1 which indicates time alloc. to the "best" planner
target_net = DQN(128, 128, n_actions + 1).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = opt.Adam(policy_net.parameters())
# what optimizer? # switched to ADAM for this project
# optimizer = opt.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)
steps_done = 0


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    number_of_non_final_states = torch.sum(non_final_mask)

    non_final_next_states = [s for s in batch.next_state if s is not None]
    non_final_next_state_batch = State(*zip(*non_final_next_states))
    # non_final_next_states = torch.tensor(non_final_next_states)
    # state_batch = [
    #     torch.cat(tuple(batch.state[j][i] for j in range(len(batch.state))) for i in range(len(batch.state[0])))]
    # state_batch = [s for s in batch.state] this doesnt change anything
    state_batch = State(*zip(*batch.state))  # creates a tuple of batches like above
    action_batch = torch.tensor([a[0] for a in batch.action]).reshape(1, -1)
    reward_batch = torch.cat(batch.reward)
    # print(state_batch)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    # below: get corresponding value from each output vector i in batch corrsponding to action in actionbatch at index i
    # and append them to a list --> state_action_values
    # print(*state_batch)
    state_img_batch = torch.cat(state_batch.img)
    state_max_consec_exec_batch = torch.cat(state_batch.maxConsecExecuted, dim=0).reshape((124, 17))
    state_curr_exec_batch = torch.cat(state_batch.currentlyExecuting, dim=0).reshape((124, 17))
    state_t_left_ep_batch = torch.cat(state_batch.time_left_ep).reshape((124, 1))
    # print(state_img_batch.size()) seems correct
    print("start time")
    nn_out_p, nn_out_t = policy_net(state_img_batch, None, state_max_consec_exec_batch, state_curr_exec_batch,
                                    state_t_left_ep_batch)
    print("finish time")
    state_action_values = nn_out_p.gather(1, action_batch.reshape(124, 1))
    state_time_alloc_Values = nn_out_t
    # for i in range(0, len(action_batch)):
    #     state_action_values.append(nn_output_vectors[i][0][action_batch[i][0].item()])
    # state_action_values = torch.stack(state_action_values)  # list of tensors to tensor

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(len(non_final_next_states), device=device)
    next_state_Values = torch.zeros(BATCH_SIZE, device=device)
    next_planner_time = torch.zeros((BATCH_SIZE, 1), device=device)
    # print(target_net(non_final_next_states))
    nxt_state_img_batch = torch.cat(non_final_next_state_batch.img)
    nxt_state_max_consec_exec_batch = torch.cat(non_final_next_state_batch.maxConsecExecuted, dim=0).reshape((
        number_of_non_final_states, 17))
    nxt_state_curr_exec_batch = torch.cat(non_final_next_state_batch.currentlyExecuting, dim=0).reshape((
        number_of_non_final_states, 17))
    nxt_state_t_left_ep_batch = torch.cat(non_final_next_state_batch.time_left_ep).reshape((
        number_of_non_final_states, 1))
    targ_out_p, targ_out_t = target_net(nxt_state_img_batch, None, nxt_state_max_consec_exec_batch,
                                        nxt_state_curr_exec_batch, nxt_state_t_left_ep_batch)
    for i in range(len(targ_out_p)):
        next_state_values[i] = torch.max(targ_out_p[i])

    next_state_Values[non_final_mask] = next_state_values
    next_planner_time[non_final_mask] = targ_out_t
    # print(next_state_Values)
    # TODO google reinforcement learning incorperate time (continous output value) into loss???
    # Compute the expected Q values
    expected_state_action_values = (next_state_Values.reshape(124, 1) * GAMMA) + reward_batch
    expected_Times = (next_planner_time * GAMMA) + reward_batch  # ???
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    criterion_t = nn.SmoothL1Loss()
    loss = criterion(state_action_values.unsqueeze(1), expected_state_action_values.unsqueeze(1))
    loss_t = criterion_t(state_time_alloc_Values.unsqueeze(1).reshape(124, 1), expected_Times.unsqueeze(1).reshape(124, 1))
    # Optimize the model
    optimizer.zero_grad()
    print("start backpropogate")
    loss.backward(retain_graph=True)
    loss_t.backward()
    print("finish backpropogate")
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def reward(taskIndex, actionNo, actionT):
    minTimeReq = trainingSet.iloc[taskIndex][actionNo + 1]
    # describing the amount of time to finish task given action choice
    # actionNo.item+1 because first column is name
    if minTimeReq <= actionT.item():
        return torch.tensor([[1]], dtype=torch.int), True
    else:
        return torch.tensor([[-1]], dtype=torch.int), False


process = T.Compose([T.ToPILImage(),
                     # T.Resize(128, interpolation=Image.CUBIC),
                     T.ToTensor()])


def select_action(select_action_State):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was found
            planner_vector, best_planner_runtime = policy_net(*select_action_State)
            return planner_vector.max(1)[1].view(1, 1), best_planner_runtime
    else:
        timeAlloc = random.random() * select_action_State[4]  # random allocation between 0 and remaining time
        actionNo = torch.tensor([[random.randrange(n_actions)]], device=device)
        return actionNo, torch.tensor([[timeAlloc]])


num_episodes = len(trainingSet) * 2
# num_episodes should not be equal to number of tasks necessarilly (in our case 2x) + random decision
taskFolderLoc = "C:/Users/TIM/PycharmProjects/pythonTestPyTorch/IPC-image-data-master/grounded/"
for i_episode in range(num_episodes):
    time_left_ep = 300
    maxConsecExecuted = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    current_task_index = math.floor(random.random() * len(trainingSet))  # pick a random task in training set
    currentTaskName = trainingSet.iloc[current_task_index][0]
    currentTaskLoc = taskFolderLoc + currentTaskName + '-bolded-cs.png'
    img = read_image(currentTaskLoc)
    # convert image to Tensor below for CNN
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    img = process(img).unsqueeze(0)
    state = (img, current_task_index, maxConsecExecuted, currentlyExecuting,
             torch.tensor([time_left_ep]))  # store task_index to calc. reward
    last_actionNumber = None
    same_action = False
    t = 0
    while t <= time_left_ep - 5:  # to prevent infinitely allocating smaller timeslots
        # Select and perform an action
        action = select_action(state)
        actionNumber = action[0].item()
        actionTime = action[1]
        if last_actionNumber == actionNumber:  # to update consecutive times below
            same_action = True
        rewardVal, done = reward(state[1], actionNumber, actionTime)
        if not done:
            time_left_ep -= actionTime
            if same_action:
                currentlyExecuting[actionNumber] += actionTime
            else:
                currentlyExecuting = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                currentlyExecuting[actionNumber] = currentlyExecuting[actionNumber] + actionTime
            max_index = max(range(len(currentlyExecuting)), key=currentlyExecuting.__getitem__)
            maxConsecExecutedNext = state[2].clone()
            if maxConsecExecutedNext[max_index] < currentlyExecuting[max_index]:
                maxConsecExecutedNext[max_index] = currentlyExecuting[max_index]
                # next_state = (img, current_task_index, maxConsecExecutedNext, currentlyExecuting, torch.tensor([
                # time_left_ep]))

            # create new state
            next_state = (
            img, current_task_index, maxConsecExecutedNext, currentlyExecuting, torch.tensor([time_left_ep]))

        else:
            # next state is final
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, rewardVal)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        print("update target network...")
        target_net.load_state_dict(policy_net.state_dict())

print('Completed Training...')

# TODO verify results on the test set
