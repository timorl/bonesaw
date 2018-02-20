#!/usr/bin/env python3

import os
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from mannequin import *
from mannequin.basicnet import *
from mannequin.gym import *

def build_agent():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 1)
    normalize= RunningNormalize(shape=(2,))
    def policy(obs):
        obs = normalize.apply(obs)
        actions, _ = model.evaluate(obs)
        return actions

    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.01, # 0.05 / 0.01 / 0.02
        horizon=20, # 40 / 10 / 20 / 5
    )
    def sgd_step(traj):
        traj = traj.modified(observations=normalize)
        model.load_params(opt.get_value())
        quiet_actions, backprop = model.evaluate(traj.o)
        grad = ((traj.a - quiet_actions).T * traj.r.T).T
        opt.apply_gradient(backprop(grad))

    def randomize_policy():
        model.load_params(opt.get_value()+((np.random.randn(model.n_params)*0.2)))

    model.randomize_policy = randomize_policy
    model.policy = policy
    model.sgd_step = sgd_step
    return model

def build_classifier():
    model = SimplePredictor(2, 2, classifier=True, normalize_inputs=True)
    return model

def save_plot(file_name, classifier, trajs, *,
        xs=lambda t: t.o[:,0],
        ys=lambda t: t.o[:,1],
        color=lambda t: ["b", "r"][np.argmax(t.a[0])]):
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
        classifier.predict(coords)[:,1].reshape(11, 11).T[::-1,:],
        zorder=0, aspect="auto", vmin=0.0, vmax=1.0,
        cmap="gray", interpolation="bicubic",
        extent=[np.min(coords[:,0]), np.max(coords[:,0]),
            np.min(coords[:,1]), np.max(coords[:,1])]
    )
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def curiosity(world):
    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent()
    classifier = build_classifier()

    rewNormalize = RunningNormalize(horizon=10)
    curNormalize = RunningNormalize(horizon=10)

    forplot = []
    for ep in range(2000):
        agent.randomize_policy()
        agent_traj = episode(world, agent.policy, max_steps=200)
        generated_traj = Trajectory(
                list(zip(
                    np.random.randn(len(agent_traj)) * 0.9 - 0.3,
                    np.random.randn(len(agent_traj)) * 0.07)),
                [[1,0]]*len(agent_traj))

        tagged_traj = Trajectory(agent_traj.o, [[0,1]]*len(agent_traj))
        classifier_traj = Trajectory.joined(tagged_traj, generated_traj)

        classifier.sgd_step(classifier_traj, lr=0.001)

        agent_traj_curio = agent_traj.modified(
                rewards=lambda r: classifier.predict(agent_traj.o)[:,0]
        )

        forplot += [tagged_traj]
        if len(forplot) >= 10:
            save_plot(
                log_dir + "/%04d.png" % (ep + 1),
                classifier, forplot
            )
            forplot = []

        print(bar(np.mean(agent_traj_curio.r), 1.0))
        print(bar(np.sum(agent_traj.r), 200.))

        agent_traj_curio = agent_traj_curio.discounted(horizon=200)
        agent_traj_curio = agent_traj_curio.modified(rewards=curNormalize)
        agent_traj = agent_traj.discounted(horizon=200)
        agent_traj = agent_traj.modified(rewards=rewNormalize)
        agent_traj = agent_traj.modified(rewards=np.tanh)
        real_weight=0.5
        curio_weight=0.5
        agent_traj = agent_traj.modified(
                rewards=lambda r: (real_weight*r + curio_weight*agent_traj_curio.r)
        )
        agent.sgd_step(agent_traj)

def run():
    world = gym.make("MountainCarContinuous-v0")

    if len(sys.argv) >= 2:
        agent = build_agent()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
