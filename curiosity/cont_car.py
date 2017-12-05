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
    def policy(obs):
        actions, _ = model.evaluate(obs)
        return actions
    model.policy = policy
    return model

def build_classifier():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 2)
    def probabilities(obs):
        probs, _ = model.evaluate(obs)
        return softmax(probs)
    model.probabilities = probabilities
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
        classifier.probabilities(coords)[:,1].reshape(11, 11).T[::-1,:],
        zorder=0, aspect="auto", vmin=0.0, vmax=1.0,
        cmap="gray", interpolation="bicubic",
        extent=[np.min(coords[:,0]), np.max(coords[:,0]),
            np.min(coords[:,1]), np.max(coords[:,1])]
    )
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def softmax(v):
    v = v.T
    v = np.exp(v - np.amax(v, axis=0))
    v /= np.sum(v, axis=0)
    return v.T


def curiosity(world):
    world = UnboundedActions(world)
    memory = []

    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent()
    agent_opt = Adams(
        np.random.randn(agent.n_params),
        lr=0.000015,
        horizon=2
    )

    classifier = build_classifier()
    classifier_opt = Adams(
        np.random.randn(classifier.n_params),
        lr=0.000005,
        horizon=10
    )

    normalize = RunningNormalize(horizon=10)

    for ep in range(1000):
        agent.load_params(agent_opt.get_value()*((np.random.rand(agent.n_params)*0.1)+0.95))
        classifier.load_params(classifier_opt.get_value())

        agent_traj = episode(world, agent.policy)
        new_memory = Trajectory(agent_traj.o, [[1,0]]*len(agent_traj))
        memory.append(new_memory)

        tagged_traj = Trajectory(agent_traj.o, [[0,1]]*len(agent_traj))
        remembered_traj = memory[np.random.choice(len(memory))]
        classifier_traj = Trajectory.joined(tagged_traj, remembered_traj)

        classifier_logits, classifier_backprop = classifier.evaluate(classifier_traj.o)
        grad = (classifier_traj.a - softmax(classifier_logits))
        classifier_opt.apply_gradient(classifier_backprop(grad))

        agent_traj = agent_traj.modified(
            rewards=lambda r: np.max(classifier.probabilities(agent_traj.o)[:,1])
        )

        if ep % 50 == 0:
            save_plot(
                log_dir + "/%04d.png" % (ep + 1),
                classifier, [tagged_traj, remembered_traj]
            )

        print(bar(agent_traj.r[0], 1.0))

        agent_traj = agent_traj.modified(rewards=normalize)
        agent.load_params(agent_opt.get_value())
        agent_quiet_actions, agent_backprop = agent.evaluate(agent_traj.o)
        grad = ((agent_traj.a - agent_quiet_actions).T * agent_traj.r.T).T
        agent_opt.apply_gradient(agent_backprop(grad))

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
