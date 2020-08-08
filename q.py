import argparse
import torch
import wandb

import random
import numpy as np

from env import Env
from models import Model, CategoricalPolicy, DeterministicPolicy, Value, QNetwork
from storage import Storage


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def explore(timesteps, env, storage):
    s = env.reset()
    for _ in range(timesteps):
        a = env.random_action()
        s2, c, done = env.step(a)
        storage.store((s, a, c, s2, done))
        s = env.reset() if done else s2


def Q_update(policy, Q, storage, batch_size):
    s, a, c, s2, done = storage.sample(batch_size)
    m = 1 - done

    # improve Q function estimator
    with torch.no_grad():
        q_target = c + 0.99 * m * Q.target(s2, policy.target(s2))
        #! q_target = c + m * Q.target(s2, policy.target(s2))
    q_loss = ((q_target - Q(s, a)) ** 2).mean()
    Q.minimize(q_loss)

    # improve policy
    policy_loss = Q(s, policy(s)).mean()
    policy.minimize(policy_loss)

    # update target networks
    Q.soft_update_target()
    policy.soft_update_target()


def train(policy_class, env_name, num_timesteps, lr, noise, batch_size, vis_iter, seed=0, log=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # create env and models
    env = Env(env_name, seed=seed)

    n_s = env.state_dim()
    n_a = env.action_dim()

    policy = Model(policy_class, lr, n_s, n_a, target=True)
    Q = Model(QNetwork, lr, n_s, n_a, target=True)

    storage = Storage(1e6)

    # add random transitions to replay buffer
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
            a = policy(s)
            a = (a + torch.randn_like(a) * noise).clamp(-2., 2.)
        s2, c, done = env.step(a)
        storage.store((s, a, c, s2, done))

        # cost bookkeeping
        ep_cost += c.item()

        # update models
        Q_update(policy, Q, storage, batch_size)

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
    parser.add_argument('--noise', type=float, default=0.15)
    args = parser.parse_args()

    # for seed in [2542, 7240, 1187, 2002, 2924]:
    for seed in [7329, 9643, 6541, 6563, 2709, 6530, 3082, 1706, 3464, 3132, 33, 9348, 1539, 8655, 5601, 4295, 4525, 4767, 3065, 3094]:
        wandb.init(project='Pendulum', group='Q-Learning', name=str(seed), reinit=True)
        train(policy_class=DeterministicPolicy, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, noise=args.noise, batch_size=args.batch, vis_iter=200, seed=seed, log=True)
        wandb.join()
