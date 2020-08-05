import torch
import gym


class Env:
    def __init__(self, name, seed=0):
        self.env = gym.make(name)
        self.env.seed(seed)

    def reset(self):
        return torch.FloatTensor(self.env.reset())

    def step(self, a):
        s, r, done, _ = self.env.step(a.numpy())
        # return cost instead of reward
        return torch.FloatTensor(s), -torch.FloatTensor([r]), torch.FloatTensor([done])
