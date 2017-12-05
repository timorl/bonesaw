#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy
from models import Input, Affine, LReLU, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, retrace, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

def convert_traj(traj, pred, class_id):
    r = 1.+(110.-len(traj))/90.
    if r < 0.1:
        result = [(o, a, p[class_id]*0.1 + r*0.9) for (o, a, _), p in zip(traj,pred)]
    else:
        result = [(o, a, r) for (o, a, _) in traj]
    return result

def learn_from_classifier(classifier, trajs, class_id):
    preds = retrace(trajs, model=classifier)
    return [convert_traj(traj, pred, class_id) for traj, pred in zip(trajs, preds)]

def carr():
    carr = Input(2)
    carr = Affine(carr, 32)
    carr = LReLU(carr)
    carr = Affine(carr, 3)
    return Softmax(carr)

def tag_traj(traj, tag):
    return [(t[0], tag, 1.) for t in traj]

def plot_tagged_trajs(trajs):
    COLORS = ["blue", "red", "green"]
    plt.ion()
    plt.clf()
    plt.grid()
    plt.gcf().axes[0].set_ylim([-0.075,0.075])
    plt.gcf().axes[0].set_xlim([-1.25,0.5])
    for traj in trajs:
        tag = traj[0][1]
        xs, ys = [], []
        for (x,y), _, _ in traj:
            xs.append(x)
            ys.append(y)
        plt.plot(xs, ys, color=COLORS[np.argmax(tag)])
    plt.pause(0.01)

def run():
    classifier = Input(2)
    classifier = Affine(classifier, 16)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    world = Gym("MountainCar-v0")
    world = StochasticPolicy(world)

    curCarr = carr()
    curCarr.load_params(np.random.randn(curCarr.n_params))
    oldTrajs = world.trajectories(curCarr, 800)

    def train_one(carrOpt):
        nonlocal oldTrajs
        classOpt = Adam(
            np.random.randn(classifier.n_params) * 1.,
            lr=0.5,
            memory=0.9,
        )
        if carrOpt == None:
            carrOpt = Adam(
                np.random.randn(curCarr.n_params),
                lr=0.10,
                memory=0.5,
            )
        curScore = 0.
        curAccuracy = 0.
        for i in range(250):
            classifier.load_params(classOpt.get_value())
            curCarr.load_params(carrOpt.get_value())

            oldTrajIdx = np.random.choice(len(oldTrajs), size=50)
            trajs = [oldTrajs[i] for i in oldTrajIdx]
            trajs += world.trajectories(curCarr, 50)
            trajsForClass = [tag_traj(traj, [1,0]) for traj in trajs[:50]]
            trajsForClass += [tag_traj(traj, [0,1]) for traj in trajs[50:]]
            plot_tagged_trajs(trajsForClass)
            accTrajs = accuracy(trajsForClass, model=classifier)
            print_reward(accTrajs, max_value=1.0, episode=np.mean, label="Cla reward: ")
            curAccuracy = np.mean(get_rewards(accTrajs, episode=np.mean))
            if curAccuracy > 1.-i/500:
                break

            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)
            trajs2 = learn_from_classifier(classifier, trajs[50:], 1)
            print_reward(trajs2, max_value=1.0, episode=np.max, label="Car reward: ")
            curScore = np.mean(get_rewards(trajs2, episode=np.max))
            trajs2 = replace_rewards(trajs2, episode=np.max)
            trajs2 = normalize(trajs2)
            grad2 = policy_gradient(trajs2, policy=curCarr)
            carrOpt.apply_gradient(grad2)
            if i % 10 == 0:
                print("%d episodes in."%i)
        oldTrajs += world.trajectories(curCarr, 800)
        world.render(curCarr)
        if curScore > 0.11:
            return carrOpt
        else:
            return None
    theCarOpt = None
    for i in range(10):
        print("Teaching agent %d."%i)
        theCarOpt = train_one(theCarOpt)

if __name__ == "__main__":
    run()
