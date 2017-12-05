#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist, Accuracy, StochasticPolicy
from models import Input, Affine, LReLU, Softmax
from trajectories import policy_gradient, print_reward
from optimizers import Adams

def run():
    model = Input(28, 28)
    model = Affine(model, 128)
    model = LReLU(model)
    model = Affine(model, 10)
    model = Softmax(model)

    train_world = StochasticPolicy(Accuracy(Mnist()))

    opt = Adams(
        np.random.randn(model.n_params),
        lr=0.00002,
        memory=0.99
    )

    for i in range(600):
        model.load_params(opt.get_value())
        trajs = train_world.trajectories(model, 128)
        print_reward(trajs, max_value=1)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

if __name__ == "__main__":
    run()
