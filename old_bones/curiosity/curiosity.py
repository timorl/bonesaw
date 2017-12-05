#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from models import Input, Affine, LReLU, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

def gauss_observation(agents):
    agent = np.random.choice(agents)
    (center,) = agent.outputs([None])
    obs = center + np.random.randn(2)*0.05
    obs += 1001.
    obs = np.abs(obs)
    obs %= 2.
    obs -= 1.
    return obs

def learn_from_classifier(classifier, trajs, class_id):
    obs = [o for ((o, _, _),) in trajs]
    pred = classifier.outputs(obs)
    return [[(None, o, p[class_id])] for o, p in zip(obs, pred)]

def run():
    classifier = Input(2)
    classifier = Affine(classifier, 16)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    gausses = [Constant(2)]
    gausses[0].load_params([0.,0.])

    plt.ion()
    def train_one():
        gaussOpt = Adam(
            [0.,0.],
            lr=0.010,
            memory=0.5,
        )
        classOpt = Adam(
            np.random.randn(classifier.n_params) * 0.1,
            lr=0.5,
            memory=0.99
        )
        gaussCenterer =Constant(2)
        gausses.append(gaussCenterer)
        curAccuracy = 0.
        while curAccuracy < 0.98:
            classifier.load_params(classOpt.get_value())
            gaussCenterer.load_params(gaussOpt.get_value())

            trajs = [[(gauss_observation(gausses[:-1]), [1,0], 1.)] for _ in range(500)]
            trajs += [[(gauss_observation(gausses[-1:]), [0,1], 1.)] for _ in range(500)]
            accTrajs = accuracy(trajs, model=classifier)
            print_reward(accTrajs, max_value=1.0)
            accs = [traj[0][2] for traj in accTrajs]
            curAccuracy = np.mean(accs)

            grad = policy_gradient(trajs, policy=classifier)
            classOpt.apply_gradient(grad)
            trajs2 = learn_from_classifier(classifier, trajs[500:], 1)
            trajs2 = normalize(trajs2)
            grad2 = policy_gradient(trajs2, policy=gaussCenterer)
            gaussOpt.apply_gradient(grad2)
            plt.clf()
            plt.grid()
            plt.gcf().axes[0].set_ylim([-1,1])
            plt.gcf().axes[0].set_xlim([-1,1])
            x, y = zip(*[o for ((o, _, _),) in trajs[:500]])
            plt.scatter(x, y, color="blue")
            x, y = zip(*[o for ((o, _, _),) in trajs[500:]])
            plt.scatter(x, y, color="red")
            plt.pause(0.01)
    for i in range(10):
        print("Teaching agent %d."%i)
        train_one()
    plt.pause(10000000000000.)

if __name__ == "__main__":
    run()
