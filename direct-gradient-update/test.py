import torch
from torch.autograd import grad
import matplotlib.pyplot as plt

# funtion we're optimizing
def f(x):
    # return (x ** 3) - (x ** 2) + 10
    return (x.log() ** 2).mean()

# variable and its optimizer
x = torch.FloatTensor([2, 2])
x.requires_grad = True
optim = torch.optim.SGD([x], lr=5e-1)

# history of points in the iterative process
x_hist = [x.detach().tolist()]

# optimization step
for _ in range(100):
    # calculate loss
    out = f(x)
    # grads = out
    grads = grad(out, x, torch.ones(out.shape), create_graph=True)[0]
    loss = (grads ** 2).mean()

    # take minimizing step
    optim.zero_grad()
    loss.backward()
    optim.step()

    x_hist.append(x.detach().tolist())

# plot results
xs, ys = zip(*x_hist)
plt.scatter(xs, ys, c=range(len(xs)))
plt.show()
