import argparse
import torch
import wandb

from env import Env
from models import CategoricalPolicy, DeterministicPolicy, Value, QNetwork
from storage import Storage


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def Q_update(storage, policy, Q, batch_size):
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


def train(policy_class, env_name, num_timesteps, lr, batch_size, vis_iter, seed=0, log=False):
    torch.manual_seed(seed)

    # create env and models
    env = Env(env_name, seed=seed)

    n_s = env.state_dim()
    n_a = env.action_dim()

    policy = policy_class(lr, n_s, n_a, target=True).to(device)
    Q = QNetwork(lr, n_s, n_a, target=True).to(device)

    storage = Storage(1e6)

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
            a = (a + torch.randn_like(a) * 0.5).clamp(-2., 2.)
        s2, c, done = env.step(a)
        storage.store((s, a, c, s2, done))

        # cost bookkeeping
        ep_cost += c.item()

        # update models
        Q_update(storage, policy, Q, batch_size)

        # transition to next state + cost bookkeeping
        if done:
            last_ep_cost = ep_cost
            ep_cost = 0
            s = env.reset()
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
    parser.add_argument('--timesteps', type=int, default=1e5)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()

    for seed in [2542, 7240, 1187, 2002, 2924]:
        wandb.init(project='Pendulum', group='Q-Learning', name=str(seed), reinit=True)
        train(policy_class=DeterministicPolicy, env_name='Pendulum-v0', num_timesteps=args.timesteps, lr=args.lr, batch_size=args.batch, vis_iter=200, seed=seed, log=False)
        train(update=update, num_episodes=args.episodes, samples=args.samples, lr=args.lr, vis_iter=10, seed=seed, log=True)
        wandb.join()