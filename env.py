import torch
import numpy as np

import gym
import pybulletgym
from gym import spaces
from gym.utils import seeding


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Env:
    def __init__(self, name, seed):
        # self.env = PendulumEnv()
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
        # s, r, done, _ = self.env.step(a)
        s, r, done, _ = self.env.step(a.cpu().numpy())
        # I use costs instead of rewards
        c = -r
        return torch.FloatTensor(s).to(device), torch.FloatTensor([c]).to(device), torch.FloatTensor([done]).to(device)

    def __getattr__(self, k):
        return getattr(self.env, k)


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


def cost_fn(s, a):
    '''
    s = [cos θ, sin θ, θ dot]
    a ∈ [-max_torque, max_torque]
    '''
    if isinstance(s, list):
        return angle_normalize(np.arccos(s[0])) ** 2 + (.1 * (s[2] ** 2)) + (.001 * (a ** 2))
    else:
        l1 = angle_normalize(np.arccos(s[:, 0].unsqueeze(-1))) ** 2
        l2 = (.1 * (s[:, 2].unsqueeze(-1) ** 2))
        l3 = (.001 * (a ** 2))
        return l1 + l2 + l3
    #! should be able to use np.arcsin(s[1]) too...


class PendulumEnv:
    def __init__(self, g=10.0):
        self.max_speed = 8
        self.max_torque = 2.
        self.dt = .05
        self.g = g
        self.m = 1.
        self.l = 1.
        self.viewer = None

        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque,
            high=self.max_torque, shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = torch.clamp(u, -self.max_torque, self.max_torque)[0]

        # reward
        cost = cost_fn([np.cos(th), np.sin(th), thdot], u)

        # dynamics
        with torch.no_grad():
            u = u.cpu().numpy()
            newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
            newth = th + newthdot * dt
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        # don't let episodes last too long
        self.timesteps += 1
        done = self.timesteps >= 200

        self.state = np.array([newth, newthdot])
        return self._get_obs(), cost, done, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.timesteps = 0
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
