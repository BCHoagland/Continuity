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


class DDPG_no_target:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a)
        self.Q = Model(QNetwork, lr, n_s, n_a)

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
            q_target = c + 0.99 * m * self.Q(s2, self.policy(s2))
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)


class HJB:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(DeterministicPolicy, lr, n_s, n_a, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

    def interact(self, s, env):
        a = self.policy(s)
        s2, c, done = env.step(a)
        a = (a + torch.randn_like(a) * 0.15).clamp(-2., 2.)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve Q function estimator
        s_grad, a_grad = batch_grad(self.Q, s, a)
        with torch.no_grad():
            future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-self.policy.target(s), a_grad)
            # future = batch_dot(s2-s, s_grad) + batch_dot(self.policy.target(s2)-a, a_grad)
            q_target = c + self.Q.target(s,a) + m * 0.99 * future
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        policy_loss = self.Q(s, self.policy(s)).mean()
        self.policy.minimize(policy_loss)

        # update target networks
        self.Q.soft_update_target()
        self.policy.soft_update_target()


class HJB_Val:
    def create_models(self, lr, n_s, n_a):
        self.μ = Model(DeterministicPolicy, lr, n_s, n_a)
        self.V = Model(Value, lr, n_s, target=True)
        self.f = Model(Dynamics, lr, n_s, n_a)

    def interact(self, s, env):
        a = self.μ(s)
        s2, c, done = env.step(a)
        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve value function estimator
        V_grad = batch_grad(self.V, s)
        with torch.no_grad():
            # v_target = c + m * batch_dot(s2 - s, V_grad)
            v_target = c + 0.99 * m * self.V.target(s2)
        v_loss = ((self.V(s) - v_target) ** 2).mean()
        self.V.minimize(v_loss)

        # improve dynamics model
        f_loss = ((self.f(s, a) - (s2 - s)) ** 2).mean()
        self.f.minimize(f_loss)

        # improve policy
        from env import cost_fn

        V_grad = batch_grad(self.V, s)
        f = self.f(s, self.μ(s))
        s.requires_grad = False
        adv = (cost_fn(s, self.μ(s)) + batch_dot(f, V_grad)).mean()
        self.μ.minimize(adv)


class RelativeQ:
    def create_models(self, lr, n_s, n_a):
        self.policy = Model(RelativePolicy, lr, n_s, n_a, target=True)
        self.Q = Model(QNetwork, lr, n_s, n_a, target=True)

        self.last_a = None

    def interact(self, s, env):
        last_a = torch.FloatTensor(env.action_space.sample()) if self.last_a is None else self.last_a

        a = last_a + self.policy(s, last_a)
        s2, c, done = env.step(a)
        a = (a + torch.randn_like(a) * 0.15).clamp(-2., 2.)

        self.last_a = a

        return s, a, c, s2, done

    def update(self, storage, batch_size):
        s, a, c, s2, done = storage.sample(batch_size)
        m = 1 - done

        # improve Q function estimator
        with torch.no_grad():
            q_target = c + m * 0.99 * self.Q.target(s2, a + self.policy.target(s2, a))
        q_loss = ((q_target - self.Q(s, a)) ** 2).mean()
        self.Q.minimize(q_loss)

        # improve policy
        s_grad, a_grad = batch_grad(self.Q, s, a)
        f = s2 - s

        obj = (c + batch_dot(f, s_grad) + batch_dot(self.policy(s, a) - a, a_grad)).mean()
        self.policy.minimize(obj)

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


#! ARE THE OUTPUTS FROM THE POLICY IN THE RIGHT RANGE IN THE FIRST PLACE? SHOULD I BE USING TANH NON-LINEARITIES?

#! TRY FIXING THE INITIAL ANGLE

#! TRY RENDERING THE PROBLEM

#! PRINT OUT VALUE AND DYNAMICS ERRORS AGAIN

#! USE TRUE DYNAMICS

#! TRY LIMITING REPLAY BUFFER
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--timesteps', type=float, default=3e4)
    parser.add_argument('--batch', type=int, default=128)
    # parser.add_argument('--actors', type=int, default=8)
    # parser.add_argument('--noise', type=float, default=0.15)
    args = parser.parse_args()

    # python ddpg.py --timesteps 1e4
    for seed in [7329, 9643, 6541, 5563, 6329, 8643, 3541, 4563, 1329, 2643]:
        # wandb.init(project='Pendulum', group='DDPG', name=str(seed), reinit=True)
        # train(algo=DDPG, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        # wandb.join()

        wandb.init(project='Pendulum', group='HJB_reduce', name=str(seed), reinit=True)
        train(algo=HJB, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        wandb.join()

    # for seed in [7329, 9643, 6541, 6563]:
        # wandb.init(project='Pendulum', group='DDPG', name=str(seed), reinit=True)
        # train(algo=DDPG, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        # wandb.join()

        # wandb.init(project='Pendulum', group='HJB', name=str(seed), reinit=True)
        # train(algo=HJB, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        # wandb.join()

        # wandb.init(project='Pendulum', group='HJB2', name=str(seed), reinit=True)
        # train(algo=HJB2, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        # wandb.join()
