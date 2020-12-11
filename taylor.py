import torch
import wandb

import random
import numpy as np

from env import Env
from models import Model, CategoricalPolicy, DeterministicPolicy, RelativePolicy, Value, QNetwork, Dynamics
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

        # improve Q function estimator
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            q_target = c + 0.99 * m * self.Q.target(s2, self.policy.target(s2))
            taylor_future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-a, a_grad)
            taylor_target = c + 0.99 * m * taylor_future
        mse = ((q_target - self.Q(s, a)) ** 2).mean()
        taylor_reg = ((taylor_target - self.Q(s,a)) ** 2).mean()

        q_loss = mse + (self.taylor_coef * taylor_reg)
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


class hjbAgent:
    def __init__(self, taylor_coef):
        pass

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

        # improve Q function estimator
        s_grad, a_grad = batch_grad(self.Q.target, s, a)
        with torch.no_grad():
            # future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-self.policy.target(s), a_grad)
            future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-a, a_grad)
            q_target = c + self.Q.target(s,a) + m * 0.99 * future
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


class hjbGreedyAgent:
    def __init__(self, taylor_coef):
        pass

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

        # improve Q function estimator
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-a, a_grad)
            q_target = c + self.Q.target(s,a) + m * 0.99 * future
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

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
                print(f'Step: {step} | Cost: {last_ep_cost}')


if __name__ == '__main__':
    env = 'HopperPyBulletEnv-v0'

    defaults = dict(
        env = env,
        seed = 3458,
        lr = 3e-4,
        noise = 0.15,
        timesteps = 2e6,
        batch = 128,
        taylor = 0.1
    )

    # wandb.init(project=f'Continuity', group=f'{env}', config=defaults)
    # config = wandb.config

    # * for taylor_coef sweeps
    # for seed in defaults['seed']:
    #     wandb.init(project=f'Continuity', group=f'{env}', name=f'{seed}-{taylor}', config=defaults, reinit=True)
    #     config = wandb.config
    #     train(algo=Agent, env_name=config.env, num_timesteps=config.timesteps, lr=config.lr, noise=config.noise, batch_size=config.batch, vis_iter=200, seed=seed, log=True, taylor_coef=config.taylor)

    # * for seed sweeps
    wandb.init(project=f'HJB-Greedy-Party', group=f'{env}', config=defaults)
    config = wandb.config
    train(algo=hjbGreedyAgent, env_name=config.env, num_timesteps=config.timesteps, lr=config.lr, noise=config.noise, batch_size=config.batch, vis_iter=200, seed=config.seed, log=True, taylor_coef=config.taylor)

    # ! currently using greedy^
