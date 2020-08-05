import random
import torch
import numpy as np
from collections import deque


def fix_size(x):
    x = x.squeeze()
    while len(x.shape) < 2:
        x = x.unsqueeze(0)
    return x


class Storage:
    def __init__(self, max_size=None):
        if max_size is not None:
            self.buffer = deque(maxlen=int(max_size))
        else:
            self.buffer = deque()

    def store(self, data):
        '''stores a single group of data'''
        data = [fix_size(x) for x in data]

        batch_size = data[0].shape[0]
        for i in range(batch_size):
            transition = tuple(fix_size(data[j][i]) for j in range(len(data)))
            self.buffer.append(transition)
    
    def get(self, batch):
        n = len(self.buffer[0])
        return [torch.cat([entry[i] for entry in batch], dim=0) for i in range(n)]

    def sample(self, batch_size):
        '''return a random sample from the stored data'''

        batch_size = min(len(self.buffer), batch_size)
        batch = random.sample(self.buffer, batch_size)
        return self.get(batch)

    def get_all(self):
        return self.get(self.buffer)

    def clear(self):
        '''clear stored data'''
        self.buffer.clear()
