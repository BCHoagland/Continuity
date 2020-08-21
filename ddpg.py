import argparse
import torch
import wandb

import random
import numpy as np

from env import Env
from models import Model, CategoricalPolicy, DeterministicPolicy, Value, QNetwork, Dynamics
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


class DDPG:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

    def interact(self, s, env):
        a = self.policy(s)
        a = (a + torch.randn_like(a) * 0.15).clamp(-2., 2.)
        s2, c, done = env.step(a)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve Q function estimator
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            q_target = c + 0.99 * m * self.Q.target(s2, self.policy.target(s2))
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


class HJB:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

    def interact(self, s, env):
        a = self.policy(s)
        s2, c, done = env.step(a)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve Q function estimator
        #! ensure gradients are *not* tracked for these two \/ (unless...)
        #? can I get rid of evaluating Q(s,a) altogether by directly optimizing s_grad and a_grad????
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-self.policy.target(s), a_grad)
            q_target = c + self.Q.target(s,a) + m * 0.99 * future
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


class HJB2:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

    def interact(self, s, env):
        a = self.policy(s)
        a = (a + torch.randn_like(a) * 0.15).clamp(-2., 2.)
        s2, c, done = env.step(a)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve Q function estimator
        #! ensure gradients are *not* tracked for these two \/ (unless...)
        #? can I get rid of evaluating Q(s,a) altogether by directly optimizing s_grad and a_grad????
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-self.policy.target(s), a_grad)
            q_target = c + self.Q.target(s,a) + m * 0.99 * future
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


def train(algo, env_name, num_timesteps, lr, batch_size, vis_iter, seed=0, log=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # create env and models
    env = Env(env_name, seed=seed)

    # set up algo
    n_s = env.state_dim()
    n_a = env.action_dim()
    algo = algo()
    algo.create_models(lr, n_s, n_a)

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
            s, a, c, s2, done = algo.interact(s, env)
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
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--timesteps', type=int, default=2e4)
    parser.add_argument('--batch', type=int, default=128)
    # parser.add_argument('--actors', type=int, default=8)
    # parser.add_argument('--noise', type=float, default=0.15)
    args = parser.parse_args()

    # train(algo=HJB2, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=10, log=False)
    # quit()

    for seed in [7329, 9643, 6541, 6563]:
        # wandb.init(project='Pendulum', group='DDPG', name=str(seed), reinit=True)
        # train(algo=DDPG, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        # wandb.join()

        wandb.init(project='Pendulum', group='HJB', name=str(seed), reinit=True)
        train(algo=HJB, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        wandb.join()
