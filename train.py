import argparse
import torch
import wandb

from env import Env
from models import CategoricalPolicy, DeterministicPolicy, Value
from storage import Storage


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def batch_dot(x, y):
    vector_len = x.shape[1]
    return torch.bmm(x.view(-1, 1, vector_len), y.view(-1, vector_len, 1)).view(-1, 1)


def batch_grad(fn, inp):
    batch_size = inp.shape[0]

    inp.requires_grad = True
    out = fn(inp)
    out.backward(torch.ones(batch_size, 1).to(device))
    return inp.grad


def episodes(env, policy, n):
    storage = Storage()

    for _ in range(n):
        s = env.reset()
        while True:
            with torch.no_grad():
                a = policy(s)
            s2, c, done = env.step(a)

            storage.store((s, a, c, s2, done))

            if done:
                break
            else:
                s = s2
    
    return storage.get_all()


def train(update, env_name, num_episodes, samples, lr, vis_iter, seed=0, log=False):
    torch.manual_seed(seed)

    # create env and models
    env = Env(env_name, seed=0)

    n_s = env.state_dim()
    n_a = env.action_dim()

    policy = CategoricalPolicy(lr, n_s, n_a).to(device)
    V = Value(lr, n_s, target=False).to(device)

    if not log:
        from visualize import plot_live

    # training loop
    for ep in range(num_episodes):
        s, a, c, s2, done = episodes(env, policy, samples)
        m = 1 - done

        # calculate returns
        returns = [0] * len(c.cpu().tolist())
        discounted_next = 0
        for i in reversed(range(len(c))):
            returns[i] = c[i] + discounted_next
            # discounted_next = 0.99 * returns[i] * m[i - 1]
            discounted_next = returns[i] * m[i - 1]
        returns = torch.stack(returns)

        # improve value function estimator
        value_loss = ((returns - V(s)) ** 2).mean()
        V.minimize(value_loss)

        # calculate and normalize advantage
        if update == 'HJB':
            adv = c + batch_dot(s2 - s, batch_grad(V, s))
        elif update == 'TD':
            # adv = c + V.target(s2) - V(s)
            adv = c + V(s2) - V(s)
        elif update == 'Standard':
            adv = returns - V(s)
        else:
            raise ValueError(f'{update} is not a known update')

        adv = (adv.detach() - adv.mean()) / adv.std()

        # improve policy
        log_prob = policy.log_prob(s, a)
        obj = (adv * log_prob).mean()
        policy.minimize(obj)

        # update target networks
        # V.soft_update_target()

        # report progress
        if ep % vis_iter == vis_iter - 1:
            if log:
                wandb.log({'Average episodic cost': sum(c).cpu().item() / samples}, step=ep)
            else:
                plot_live(ep, sum(c).cpu().item() / samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--samples', type=int, default=2)
    args = parser.parse_args()

    #! test 'Standard' with lr=3e-4

    #! GPU runs might not be deterministic

    # for seed in [2542, 7240, 1187, 2002, 2924]:
    #     for update in ['HJB', 'TD']:
    #         wandb.init(project='Continuity-Experiments', group=update, name=str(seed), reinit=True)
    #         train(update=update, num_episodes=args.episodes, samples=args.samples, lr=args.lr, vis_iter=10, seed=seed, log=True)
    #         wandb.join()

    train(update='TD', env_name='CartPole-v1', num_episodes=args.episodes, samples=args.samples, lr=args.lr, vis_iter=10, seed=0, log=False)
