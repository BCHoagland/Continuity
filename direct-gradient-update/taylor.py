import torch
from torch.autograd import grad
import wandb

import random
import numpy as np

from env import Env
from models import Model, DeterministicPolicy, QNetwork
from storage import Storage


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def batch_dot(x, y):
    vector_len = x.shape[1]
    return torch.bmm(x.view(-1, 1, vector_len), y.view(-1, vector_len, 1)).view(-1, 1)


def batch_grad(fn, *inputs):
    for inp in inputs:
        inp.requires_grad = True
    out = fn(*inputs)
    out.backward(torch.ones_like(out).to(device))

    if len(inputs) == 1:
        return inputs[0].grad
    return [inp.grad for inp in inputs]


def explore(timesteps, env, storage):
    s = env.reset()
    for _ in range(timesteps):
        a = env.random_action()
        s2, c, done = env.step(a)
        storage.store((s, a, c, s2, done))
        s = env.reset() if done else s2


class Agent:
    def __init__(self, taylor_coef):
        self.taylor_coef = taylor_coef

    def create_models(self, lr, n_s, n_a, action_space):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a, action_space, device, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

    def interact(self, s, env, noise):
        a = self.policy(s)
        a = (a + torch.randn_like(a) * noise).clamp(env.action_space.low[0], env.action_space.high[0])
        s2, c, done = env.step(a)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        with torch.no_grad():
            q_target = c + m * 0.99 * self.Q.target(s2, self.policy.target(s2))
        q_loss = ((self.Q(s, a) - q_target) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve Q function estimator
        # s.requires_grad = True
        # a.requires_grad = True

        # q = self.Q(s, a)
        # s_grad = grad(q, s, torch.ones(q.shape), create_graph=True)[0]
        # a_grad = grad(q, a, torch.ones(q.shape), create_graph=True)[0]

        # bruh = c + batch_dot(s2 - s, s_grad) + batch_dot(self.policy.target(s2) - a, a_grad)
        # # bruh = c + batch_dot(s2 - s, s_grad) + batch_dot(self.policy(s2) - a, a_grad)
        # #! scaling for c? I'm pretty sure the 'c' I'm using is actually 'δc'...
        # q_loss = (bruh ** 2).mean()
        # self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


def train(algo, env_name, num_timesteps, lr, noise, batch_size, vis_iter, seed=0, log=False, taylor_coef=0.5):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # create env and models
    env = Env(env_name, seed=seed)

    # set up algo
    n_s = env.state_dim()
    n_a = env.action_dim()
    algo = algo(taylor_coef)
    algo.create_models(lr, n_s, n_a, env.action_space)

    # create storage and add random transitions to it
    storage = Storage(1e6)
    explore(10000, env, storage)

    if not log:
        from visualize import plot_live

    # training loop
    last_ep_cost = 0
    ep_cost = 0

    s = env.reset()
    for step in range(int(num_timesteps)):
        # interact with env
        with torch.no_grad():
            s, a, c, s2, done = algo.interact(s, env, noise)
        storage.store((s, a, c, s2, done))

        # cost bookkeeping
        ep_cost += c.item()

        # algo update
        algo.update(storage, batch_size)

        # transition to next state + cost bookkeeping
        if done:
            s = env.reset()
            last_ep_cost = ep_cost
            ep_cost = 0
        else:
            s = s2

        # report progress
        if step % vis_iter == vis_iter - 1:
            if log:
                wandb.log({'Average episodic cost': last_ep_cost}, step=step)
            else:
                plot_live(step, last_ep_cost)


if __name__ == '__main__':
    train(algo=Agent, env_name='Pendulum-v0', num_timesteps=1e5, lr=3e-4, noise=0.15, batch_size=128, vis_iter=200, seed=0, log=False)
