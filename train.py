import torch
import wandb

from env import Env
from models import CategoricalPolicy, Value
from storage import Storage


def batch_dot(x, y):
    vector_len = x.shape[1]
    return torch.bmm(x.view(-1, 1, vector_len), y.view(-1, vector_len, 1)).view(-1, 1)


def batch_grad(fn, inp):
    batch_size = inp.shape[0]

    inp.requires_grad = True
    out = fn(inp)
    out.backward(torch.ones(batch_size, 1))
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


def train(num_episodes, samples, vis_iter, seed=0, log=False):
    torch.manual_seed(seed)

    # create env and models
    env = Env('CartPole-v1', seed=0)
    policy = CategoricalPolicy(lr)
    V = Value(lr, target=True)

    if not log:
        from visualize import plot_live

    # training loop
    for ep in range(num_episodes):
        s, a, c, s2, done = episodes(env, policy, samples)
        m = 1 - done

        # calculate returns
        returns = [0] * len(c.tolist())
        discounted_next = 0
        for i in reversed(range(len(c))):
            returns[i] = c[i] + discounted_next
            discounted_next = 0.99 * returns[i] * m[i - 1]
        returns = torch.stack(returns)

        # improve value function estimator
        with torch.no_grad():
            # value_target = c + V.target(s2)
            value_target = returns
        value_loss = ((value_target - V(s)) ** 2).mean()
        V.minimize(value_loss)

        # calculate and normalize advantage
        if update == 'HJB':
            adv = (c + batch_dot(s2 - s, batch_grad(V, s))).detach()
        elif update == 'TD':
            adv = (c + V(s2) - V(s)).detach()
        elif update == 'Standard':
            adv = returns - V(s)
        else:
            raise ValueError(f'{update} is not a known update')

        adv = (adv - adv.mean()) / adv.std()

        # improve policy
        log_prob = policy.log_prob(s, a)
        obj = (adv * log_prob).mean()
        policy.minimize(obj)

        # report progress
        if ep % vis_iter == vis_iter - 1:
            if log:
                wandb.log({'Average episodic cost': sum(c).item() / samples}, step=ep)
            else:
                plot_live(ep, sum(c).item() / samples)


if __name__ == '__main__':
    lr = 1e-3

    for seed in [2542, 7240, 1187, 2002, 2924]:
        for update in ['HJB', 'TD']:
            wandb.init(project='Continuity-Experiments', group=update, name=str(seed), reinit=True)
            train(num_episodes=1500, samples=10, vis_iter=10, seed=seed, log=True)
            wandb.join()
