from collections import deque, namedtuple
import torchvision.transforms as T
import random

from PIL import Image

imageHeight = 128  # must be 128 or smaller
imageWidth = 128  # must be 128 or smaller

Transition = namedtuple('Transition',
                        ('state', 'action', 'done', 'next_state', 'reward')
                        )

resize = T.Compose([T.ToPILImage(),
                    T.Resize(imageWidth, interpolation=Image.CUBIC),
                    T.ToTensor()])


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
