from copy import deepcopy
import torch
import torch.nn as nn
# from torch.distributions import Normal
from torch.distributions import Categorical

n_h = 32
τ = 0.995


class Model(nn.Module):
    def __init__(self, lr, *sizes, target):
        super().__init__()

        self.optimizer = None
        self.lr = lr

        if target:
            self.target_model = self.__class__(lr, *sizes, target=False)

    def target(self, *args):
        return self.target_model(*args)

    def soft_update_target(self):
        for param, target_param in zip(self.parameters(), self.target_model.parameters()):
            target_param.data.copy_((τ * target_param.data) + ((1 - τ) * param.data))

    def _optimize(self, loss):
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def maximize(self, loss):
        self._optimize(-loss)

    def minimize(self, loss):
        self._optimize(loss)


class CategoricalPolicy(Model):
    def __init__(self, lr, n_s, n_a, target=False):
        super().__init__(lr, n_s, n_a, target=target)

        self.main = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_a)
        )
        # self.sigma = nn.Parameter(torch.zeros(n_a))

    def dist(self, s):
        # return Normal(self.main(s), self.sigma.exp())
        return Categorical(logits=self.main(s))

    def forward(self, s):
        return self.dist(s).sample()

    def log_prob(self, s, a):
        return self.dist(s).log_prob(a)


class DeterministicPolicy(Model):
    def __init__(self, lr, n_s, n_a, target=False):
        super().__init__(lr, n_s, n_a, target=target)

        self.main = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_a)
        )

    def forward(self, s):
        return self.main(s)


class Value(Model):
    def __init__(self, lr, n_s, target=False):
        super().__init__(lr, n_s, target=target)

        self.main = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, 1)
        )

    def forward(self, s):
        return self.main(s)


class QNetwork(Model):
    def __init__(self, lr, n_s, n_a, target=False):
        super().__init__(lr, n_s, n_a, target=target)

        self.main = nn.Sequential(
            nn.Linear(n_s + n_a, n_h),
            nn.ReLU(),
            nn.Linear(n_h, n_h),
            nn.ReLU(),
            nn.Linear(n_h, 1)
        )

    def forward(self, s, a):
        return self.main(torch.cat([s, a], dim=-1))
