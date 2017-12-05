#!/usr/bin/env python3

import os
import sys
import numpy as np
import gym

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, StochasticPolicy
from models import Input, Affine, LReLU, Softmax
from trajectories import policy_gradient, normalize, discount, print_reward
from optimizers import Adam

def make_env():
    env = gym.make("CartPole-v1")
    orig_step = env.step
    def step(action):
        env.unwrapped.steps_beyond_done = None
        o, r, _, i = orig_step(action)
        return o, -abs(o[0]), False, i
    env.step = step
    return env

def run():
    model = Input(4)
    model = Affine(model, 128)
    model = LReLU(model)
    model = Affine(model, 2)
    model = Softmax(model)

    world = StochasticPolicy(Gym(make_env, max_steps=500))

    opt = Adam(
        np.random.randn(model.n_params) * 0.1,
        lr=0.01
    )

    for _ in range(50):
        model.load_params(opt.get_value())

        trajs = world.trajectories(model, 16)
        print_reward(trajs, max_value=5000)

        trajs = discount(trajs, horizon=500)
        trajs = normalize(trajs)

        grad = policy_gradient(trajs, policy=model)
        opt.apply_gradient(grad)

    while True:
        world.render(model)

if __name__ == "__main__":
    run()
