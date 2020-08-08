from copy import deepcopy
import torch
import torch.nn as nn
# from torch.distributions import Normal
from torch.distributions import Categorical


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

n_h = 64
τ = 0.995


class Model:
    def __init__(self, model_type, lr, *args, target=False):
        self.model = model_type(*args).to(device)
        if target:
            self.target_model = model_type(*args).to(device)
            self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def maximize(self, loss):
        self._optimize(-loss)

    def minimize(self, loss):
        self._optimize(loss)

    def target(self, *args):
        with torch.no_grad():
            return self.target_model(*args)

    def log_prob(self, *args):
        return self.model.log_prob(*args)

    def __getattr__(self, k):
        return getattr(self.model, k)

    def __call__(self, *args):
        return self.model(*args)

    def soft_update_target(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_((τ * target_param.data) + ((1 - τ) * param.data))


class CategoricalPolicy(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()

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


class DeterministicPolicy(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_a)
        )

    def forward(self, s):
        return self.main(s)


class Value(nn.Module):
    def __init__(self, n_s):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.Tanh(),
            nn.Linear(n_h, n_h),
            nn.Tanh(),
            nn.Linear(n_h, 1)
        )

    def forward(self, s):
        return self.main(s)


class QNetwork(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()

        self.pre_state = nn.Sequential(
            nn.Linear(n_s, n_h),
            nn.ELU(),
            nn.Linear(n_h, n_h // 2),
            nn.ELU()
        )

        self.pre_action = nn.Sequential(
            nn.Linear(n_a, n_h // 2),
            nn.ELU()
        )

        self.main = nn.Sequential(
            nn.Linear(n_h, n_h),
            nn.ELU(),
            nn.Linear(n_h, 1)
        )

    def forward(self, s, a):
        s = self.pre_state(s)
        a = self.pre_action(a)
        return self.main(torch.cat([s, a], dim=-1))
