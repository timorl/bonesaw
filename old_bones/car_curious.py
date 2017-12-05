#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, ActionNoise, Cache
from models import Input, Affine, Softmax, LReLU
from trajectories import (normalize, discount, policy_gradient,
    print_reward, get_rewards, replace_rewards, retrace)
from optimizers import Adams

def build_agent():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 1)
    return model

def build_classifier():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 2)
    model = Softmax(model)
    return model

def save_plot(file_name, classifier, trajs, *,
        xs=lambda t: [o[0] for o, a, r in t],
        ys=lambda t: [o[1] for o, a, r in t],
        color=lambda t: ["b", "r"][np.argmax(t[0][1])]):
    import matplotlib.pyplot as plt
    coords = (np.mgrid[0:11,0:11].reshape(2,-1).T
        * [0.175, 0.015] - [1.25, 0.075])
    plt.clf()
    for t in trajs:
        plt.plot(
            xs(t), ys(t),
            color=color(t), alpha=0.2, linewidth=2, zorder=1
        )
    plt.imshow(
        classifier.outputs(coords)[:,1].reshape(11, 11).T[::-1,:],
        zorder=0, aspect="auto", vmin=0.0, vmax=1.0,
        cmap="gray", interpolation="bicubic",
        extent=[np.min(coords[:,0]), np.max(coords[:,0]),
            np.min(coords[:,1]), np.max(coords[:,1])]
    )
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def supervised(trajs, *, label):
    return [
        [(o, label(o, a, r), 1.0) for o, a, r in t]
        for t in trajs
    ]

def curiosity(world):
    world = ActionNoise(world, stddev=0.1)
    memory = Cache(delay=32 * 25)

    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent()
    agent_opt = Adams(
        np.random.randn(agent.n_params),
        lr=0.00015,
        memory=0.5
    )

    classifier = build_classifier()
    classifier_opt = Adams(
        np.random.randn(classifier.n_params),
        lr=0.00005,
        memory=0.9
    )

    for episode in range(1000):
        agent.load_params(agent_opt.get_value())
        classifier.load_params(classifier_opt.get_value())

        agent_trajs = world.trajectories(agent, 32)
        memory.add_trajectory(*agent_trajs)

        classifier_trajs = (
            supervised(
                memory.trajectories(None, 32),
                label=lambda *_: [1, 0]
            )
            + supervised(
                agent_trajs,
                label=lambda *_: [0, 1]
            )
        )

        grad = policy_gradient(classifier_trajs, policy=classifier)
        classifier_opt.apply_gradient(grad)

        agent_trajs = replace_rewards(
            agent_trajs,
            model=classifier,
            reward=lambda o: o[1],
            episode=lambda rs: np.square(rs) / len(rs)
        )

        save_plot(
            log_dir + "/%04d.png" % (episode + 1),
            classifier, classifier_trajs
        )

        print_reward(agent_trajs, max_value=1.0)

        agent_trajs = discount(agent_trajs, horizon=100)
        agent_trajs = normalize(agent_trajs)
        grad = policy_gradient(agent_trajs, policy=agent)
        agent_opt.apply_gradient(grad)

def run():
    world = Gym("MountainCarContinuous-v0")

    if len(sys.argv) >= 2:
        agent = build_agent()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
