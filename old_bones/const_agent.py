#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Mnist
from models import Constant
from trajectories import policy_gradient
from optimizers import Momentum

def run():
    model = Constant(10)
    world = Mnist()

    opt = Momentum(
        np.random.randn(model.n_params),
        lr=0.05,
        memory=0.9
    )

    for _ in range(100):
        model.load_params(opt.get_value())
        trajs = world.trajectories(None, 128)
        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)
        print(np.round(opt.get_value(), 2))

if __name__ == "__main__":
    run()
