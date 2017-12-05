#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld, ActionNoise, Cache
from models import Input, Affine, LReLU, Softmax, Constant
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

class Curiosity(BaseWorld):
    def __init__(self, inner, *, classifier, history_length, plot=False):
        history = Cache()
        classOpt = None

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
                plt.plot(xs, ys, color=COLORS[np.argmax(tag)], alpha=0.1)
            plt.pause(0.01)

        def remember_agent(agent):
            nonlocal history, classOpt
            history.add_trajectory(
                *inner.trajectories(agent, history_length)
            )
            classOpt = Adam(
                np.random.randn(classifier.n_params) * 1.,
                lr=0.5,
                memory=0.9,
            )

        def trajectories(agent, n):
            if classOpt is None:
                remember_agent(agent)

            oldTrajs = history.trajectories(None, n)
            innerTrajs = inner.trajectories(agent, n)

            trajsForClass = (
                [tag_traj(traj, [1,0]) for traj in oldTrajs]
                + [tag_traj(traj, [0,1]) for traj in innerTrajs]
            )

            if plot:
                plot_tagged_trajs(trajsForClass)

            classifier.load_params(classOpt.get_value())
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            curiosityTrajs = replace_rewards(
                innerTrajs,
                model=classifier,
                reward=lambda o: o[1]
            )
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.remember_agent = remember_agent

from models.BaseWrapper import BaseWrapper

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]


def carr():
    carr = Input(2)
    carr = Affine(carr, 32)
    carr = LReLU(carr)
    carr = Affine(carr, 1)
    return carr

def run():
    classifier = Input(2)
    classifier = Affine(classifier, 16)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    curCarr = carr()
    curCarr.load_params(np.random.randn(curCarr.n_params))

    world = Gym("MountainCarContinuous-v0", max_steps=500)
    world = ActionNoise(world, stddev=0.1)
    world = Curiosity(
                world,
                classifier=classifier,
                history_length=800,
                plot=True
            )

    def train_one(carrOpt):
        if carrOpt == None:
            carrOpt = Adam(
                np.random.randn(curCarr.n_params),
                lr=0.10,
                memory=0.5,
            )
        nextBreak = 5
        for i in range(250):
            curCarr.load_params(carrOpt.get_value())

            realTrajs, curiosityTrajs = world.trajectories(curCarr, 50)
            curScore = np.mean(get_rewards(realTrajs, episode=np.sum))/90.
            print_reward(realTrajs, max_value=90.0, episode=np.sum, label="Real reward:      ")
            print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
            curCuriosity = np.mean(get_rewards(curiosityTrajs, episode=np.max))
            if curCuriosity > 0.98:
                if nextBreak == 0:
                    break
                else:
                    nextBreak -= 1
            else:
                nextBreak = np.min([nextBreak+1, 5])

            realTrajs = replace_rewards(realTrajs, episode=np.sum)
            realTrajs = normalize(realTrajs)
            curiosityTrajs = replace_rewards(curiosityTrajs, episode=np.max)
            #this is stupid, we should care more(?) if the costs are to high
            realWeight = 0.001 + np.max([np.min([curScore, 0.2]), 0.])*0.998/0.2
            curiosityWeight = 1. - realWeight
            print('RWeight: %f, CWeight: %f'%(realWeight, curiosityWeight))
            trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
            trajs = normalize(trajs)
            grad = policy_gradient(trajs, policy=curCarr)
            carrOpt.apply_gradient(grad)
            if i % 10 == 0:
                print("%d episodes in."%i)
        world.remember_agent(curCarr)
        world.render(curCarr)
        if curScore > 0.01:
            return carrOpt
        else:
            return None
    theCarOpt = None
    for i in range(50):
        print("Teaching agent %d."%i)
        theCarOpt = train_one(theCarOpt)

if __name__ == "__main__":
    run()
