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
from models import Input, LSTM, Affine, Softmax
from trajectories import policy_gradient, cross_entropy, print_reward
from optimizers import Adam

def train(world, model):
    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.01,
        memory=0.9
    )

    for _ in range(1000):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 1)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

        trajs = cross_entropy(trajs, model=model)
        print_reward(trajs, episode=np.mean,
            label="Surprise/byte:", max_value=8.0)

    return model

def run():
    if len(sys.argv) < 2:
        print("Usage: imitate.py <file>")
        return

    with open(sys.argv[1], "r") as f:
        data = f.buffer.read()
    charset = set(data)
    world = Bytes(data, max_steps=100, charset=charset)
    print("Charset size: %d" % len(charset))

    model = Input(len(charset))
    model = LSTM(model)
    model = Affine(model, len(charset))
    model = Softmax(model)

    train(world, model)

    for _ in range(10):
        world.render(model)

if __name__ == "__main__":
    run()
