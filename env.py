import torch
import gym

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Env:
    def __init__(self, name, seed):
        self.env = gym.make(name)
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def state_dim(self):
        return self.env.observation_space.shape[0]

    def action_dim(self):
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]
    
    def random_action(self):
        return torch.FloatTensor(self.env.action_space.sample()).to(device)

    def reset(self):
        return torch.FloatTensor(self.env.reset()).to(device)

    def step(self, a):
        s, r, done, _ = self.env.step(a.cpu().numpy())
        # return cost instead of reward
        return torch.FloatTensor(s).to(device), -torch.FloatTensor([r]).to(device), torch.FloatTensor([done]).to(device)

    def __getattr__(self, k):
        return getattr(self.env, k)
