import torch
import gym

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Env:
    def __init__(self, name, seed, n):
        self.envs = []
        for i in range(n):
            env = gym.make(name)
            env.seed(seed)
            self.envs.append(env)


    def state_dim(self):
        return self.envs[0].observation_space.shape[0]

    def action_dim(self):
        if isinstance(self.envs[0].action_space, gym.spaces.discrete.Discrete):
            return self.envs[0].action_space.n
        else:
            return self.envs[0].action_space.shape[0]

    def reset(self):
        return torch.cat([torch.FloatTensor(env.reset()).unsqueeze(0) for env in self.envs], dim=0).to(device)

    def step(self, a):
        def unwrap(j, make_list=False):
            if make_list:
                return torch.cat([torch.FloatTensor([transitions[i][j]]).unsqueeze(0) for i in n], dim=0).to(device)    
            return torch.cat([torch.FloatTensor(transitions[i][j]).unsqueeze(0) for i in n], dim=0).to(device)

        n = range(len(self.envs))
        transitions = [self.envs[i].step(a[i].cpu().numpy()) for i in n]
        return unwrap(0), -unwrap(1, True), unwrap(2, True)

    def next_step(self, s2, done):
        s = torch.zeros_like(s2)
        for i in range(len(s2)):
            if done[i]:
                s[i] = torch.FloatTensor(self.envs[i].reset())
            else:
                s[i] = s2[i]
        return s.to(device)

    def __getattr__(self, k):
        return getattr(self.envs[0], k)
