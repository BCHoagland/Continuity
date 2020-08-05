import torch
from visdom import Visdom
import numpy as np

vis = Visdom()
wins = {}


def plot_fn(fn, fn_name):
    with torch.no_grad():
        x = torch.arange(-10., 10., 0.5).unsqueeze(1)
        y = fn(x)
    vis.line(
        X=x.squeeze(),
        Y=y.squeeze(),
        win=fn_name,
        opts=dict(
            title=fn_name
        )
    )


def plot_live(x, y, name='episodic cost'):
    global wins

    if name not in wins:
        wins[name] = vis.line(
                    X=np.array([x]),
                    Y=np.array([y]),
                    win=name,
                    opts=dict(
                        title=name
                    )
                )
        x += 1
    else:
        vis.line(
            X=np.array([x]),
            Y=np.array([y]),
            win=name,
            update='append'
        )
        x += 1
