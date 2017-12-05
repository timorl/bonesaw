#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Bytes
from models import Constant, Input, Affine, LReLU, History, Softmax
from trajectories import policy_gradient, cross_entropy, print_reward
from optimizers import Adam

def train(world, model):
    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.3,
        memory=0.9
    )

    for _ in range(20):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 100)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        trajs = cross_entropy(trajs, model=model)
        print_reward(trajs, episode=np.mean,
            label="Surprise/byte:", max_value=8.0)

def run():
    world = Bytes(b"aabbaab", max_steps=4, charset=b'abcd')

    print("\nConstant model:\n")
    model = Constant(4)
    model = Softmax(model)
    train(world, model)

    print("\nLast character:\n")
    model = Input(4)
    model = Affine(model, 4)
    model = LReLU(model)
    model = Affine(model, 4)
    model = Softmax(model)
    train(world, model)

    print("\nLast two characters:\n")
    model = Input(2, 4)
    model = Affine(model, 4)
    model = LReLU(model)
    model = Affine(model, 4)
    model = History(model, length=2)
    model = Softmax(model)
    train(world, model)

if __name__ == "__main__":
    run()
