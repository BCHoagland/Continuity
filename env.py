import torch
import gym


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Env:
    def __init__(self, name, seed=0):
        self.env = gym.make(name)
        self.env.seed(seed)

    def reset(self):
        return torch.FloatTensor(self.env.reset())

    def step(self, a):
        s, r, done, _ = self.env.step(a.cpu().numpy())
        # return cost instead of reward
        return torch.FloatTensor(s).to(device), -torch.FloatTensor([r]).to(device), torch.FloatTensor([done]).to(device)
