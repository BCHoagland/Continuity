import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HumanoidPyBulletEnv-v0')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--taylor', type=float, nargs='+', default=[0.5])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--noise', type=float, default=0.15)
    parser.add_argument('--timesteps', type=float, default=5e5)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--vis_iter', type=int, default=200)
    # parser.add_argument('--actors', type=int, default=8)
    args = parser.parse_args()

    # hyperparameter_defaults = dict(
    #     taylor = 0.1,
    # )

    # seeds = [3458, 628, 2244, 9576, 7989, 358, 6550, 1951, 2834, 5893, 6873, 9669, 7344, 6462, 8211, 7376, 9220, 7999, 7991, 2125]
    # for seed in seeds:
    #     # taylor = hyperparameter_defaults['taylor']
    #     # wandb.init(project=f'Taylor-{args.env}', name=f'{seed}-{taylor}', config=hyperparameter_defaults, reinit=True)
    #     wandb.init(project=f'Taylor-{args.env}', config=hyperparameter_defaults, reinit=True)
    #     config = wandb.config
    #     train(algo=Agent, env_name=args.env, num_timesteps=args.timesteps, lr=args.lr, noise=args.noise, batch_size=args.batch, vis_iter=args.vis_iter, seed=seed, log=True, taylor_coef=config.taylor)

    # seeds: 3458 628 2244 9576 7989 358 6550 1951 2834 5893 6873 9669 7344 6462 8211 7376 9220 7999 7991 2125
    # clear && python taylor.py --seeds 3458 628 2244 9576 7989 358 6550 1951 2834 5893 6873 9669 7344 6462 8211 7376 9220 7999 7991 2125 --taylor 0 0.1 0.25 0.5 0.75 1

    for seed in args.seeds:
        for taylor in args.taylor:
            group = str(taylor) if args.name == '' else f'{taylor} ({args.name})'
            wandb.init(project=f'Taylor-{args.env}', group=group, name=str(seed), reinit=True)
            train(algo=Agent, env_name=args.env, num_timesteps=args.timesteps, lr=args.lr, noise=args.noise, batch_size=args.batch, vis_iter=args.vis_iter, seed=seed, log=True, taylor_coef=taylor)
            wandb.join()
