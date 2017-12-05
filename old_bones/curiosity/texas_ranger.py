#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../..")

from worlds import Gym, StochasticPolicy, BaseWorld, ActionNoise, Cache
from models import Input, Affine, LReLU, Softmax, Tanh
from optimizers import Adam
from trajectories import policy_gradient, normalize, discount, print_reward, accuracy, get_rewards, replace_rewards

if "DEBUG" in os.environ:
    import sys
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

class Curiosity(BaseWorld):
    def __init__(self, inner, *, classifier, history_length, for_classifier=lambda x: x, plot=None):
        history = Cache()
        classOpt = None

        def tag_traj(traj, tag):
            return [(t[0], tag, t[2]) for t in traj]

        def remember(agent):
            nonlocal history, classOpt
            history.add_trajectory(
                *inner.trajectories(agent, history_length)
            )
            classOpt = Adam(
                np.random.randn(classifier.n_params) * 1.,
                lr=0.06,
                memory=0.9,
            )

        def trajectories(agent, n):
            if classOpt is None:
                remember(agent)

            oldTrajs = history.trajectories(None, n)
            innerTrajs = inner.trajectories(agent, n)

            trajsForClass = for_classifier(
                [tag_traj(traj, [1,0]) for traj in oldTrajs]
                + [tag_traj(traj, [0,1]) for traj in innerTrajs]
            )
            trajsForClass = replace_rewards(trajsForClass, reward=lambda r: 1.)

            if plot is not None:
                plot(trajsForClass)

            classifier.load_params(classOpt.get_value())
            grad = policy_gradient(trajsForClass, policy=classifier)
            classOpt.apply_gradient(grad)

            curiosityTrajs = replace_rewards(
                for_classifier(innerTrajs),
                model=classifier,
                reward=lambda o: o[1],
            )
            return innerTrajs, curiosityTrajs

        self.trajectories = trajectories
        self.render = inner.render
        self.remember = remember

from models.BaseWrapper import BaseWrapper

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]

def change_obs_space(trajs, changer=lambda x:x):
    return [[(changer(o, r),a,r) for o, a, r in t] for t in trajs]

def interesting_part(obs, r):
    result = (
            obs[0],
            obs[1],
            obs[2],
            obs[3],
            obs[-12],
            obs[-11],
            r
            )
    return result

STATE_SIZE = 24
ACTION_SIZE = 4
MAX_STEPS = 200

def walker():
    walker = Input(STATE_SIZE)
    walker = Affine(walker, 64)
    walker = LReLU(walker)
    walker = Affine(walker, ACTION_SIZE)
    walker = Tanh(walker)
    return walker

def run():
    classifier = Input(7)
    classifier = Affine(classifier, 32)
    classifier = LReLU(classifier)
    classifier = Affine(classifier, 2)
    classifier = Softmax(classifier)

    agent = walker()
    agent.load_params(np.random.randn(agent.n_params)*1.5)

    MAX_TRAIN_TIME = 200
    trainTimeLeft = MAX_TRAIN_TIME
    curAgentId = -1
    curMemoryId = 0
    def plot_tagged_trajs(trajs):
        nonlocal trainTimeLeft, curAgentId, curMemoryId
        COLORS = ["blue", "red"]
        plt.clf()
        plt.grid()
        plt.gcf().axes[0].set_xlim([-1.25,1.25])
        plt.gcf().axes[0].set_ylim([-1.25,1.25])
        plt.suptitle("Episode %d of agent %d, memories: %d"%(MAX_TRAIN_TIME-trainTimeLeft, curAgentId, curMemoryId))
        for traj in trajs:
            tag = traj[0][1]
            xs, ys = [], []
            for state, _, _ in traj:
                x = state[2]
                y = state[3]
                xs.append(x)
                ys.append(y)
            plt.plot(xs, ys, color=COLORS[np.argmax(tag)], alpha=0.1)
        plt.gcf().set_size_inches(10, 8)
        plt.gcf().savefig(
            "__step_a%03d_t%03d.png" %
                (curAgentId, MAX_TRAIN_TIME-trainTimeLeft),
            dpi=100
        )


    world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
    world = ActionNoise(world, stddev=0.2)
    world = Curiosity(
                world,
                classifier=classifier,
                history_length=50,
                for_classifier=lambda ts: change_obs_space(
                    ts,
                    changer=interesting_part
                    ),
                plot=plot_tagged_trajs
            )
    MAX_BOREDOM = 3
    boredom = MAX_BOREDOM

    MAX_MOTIVATION = 3
    motivation = MAX_MOTIVATION

    agentOpt = None
    lastScores = None

    def memorize():
        nonlocal boredom, curMemoryId
        print("Memorizing %d..."%curMemoryId)
        world.remember(agent)
        boredom = MAX_BOREDOM
        curMemoryId += 1
    def save_agent():
        np.save(
            "__ranger_a%03d_t%03d.npy" %
                (curAgentId, MAX_TRAIN_TIME-trainTimeLeft),
            agentOpt.get_value()
        )
    def reset_agent():
        nonlocal agentOpt, trainTimeLeft, lastScores, curAgentId, motivation
        if agentOpt is not None:
            save_agent()
        print("Resetting agent %d."%curAgentId)
        agentOpt = Adam(
            np.random.randn(agent.n_params)*1.5,
            lr=0.05,
            memory=0.9,
        )
        trainTimeLeft = MAX_TRAIN_TIME
        lastScores = [-0.4]
        curAgentId += 1
        motivation = MAX_MOTIVATION
    reset_agent()
    while True:
        agent.load_params(agentOpt.get_value())

        realTrajs, curiosityTrajs = world.trajectories(agent, 30)
        curScore = np.mean(get_rewards(realTrajs, episode=np.sum))/300.
        lastScores.append(curScore)
        lastScores = lastScores[-10:]
        scoreDev = np.std(lastScores)
        scoreMean = np.max([np.abs(np.mean(lastScores)),1.])

        curCuriosity = np.mean(get_rewards(curiosityTrajs, episode=np.max))

        print_reward(realTrajs, max_value=300.0, episode=np.sum, label="Real reward:      ")
        print_reward(curiosityTrajs, max_value=1.0, episode=np.max, label="Curiosity reward: ")
        if curCuriosity > 0.85:
            if boredom == 0:
                save_agent()
                memorize()
            else:
                boredom -= 1
        else:
            boredom = np.min([boredom+1, MAX_BOREDOM])

        if scoreDev/scoreMean < 0.010 or trainTimeLeft < 0:
            if motivation == 0:
                print("Not really learning.")
                save_agent()
                motivation = MAX_MOTIVATION
                trainTimeLeft = MAX_TRAIN_TIME
                if curScore < 0.01:
                    memorize()
                    reset_agent()
                    continue
            else:
                motivation -= 1
        else:
            motivation = np.min([motivation+1, MAX_MOTIVATION])

        realTrajs = discount(realTrajs, horizon=200)
        realTrajs = normalize(realTrajs)
        curiosityTrajs = replace_rewards(curiosityTrajs, episode=np.max)
        realWeight = np.min([scoreDev/scoreMean * 10., 0.9])
        curiosityWeight = 1. - realWeight
        trajs = combine_rewards([realTrajs, curiosityTrajs], [realWeight, curiosityWeight])
        trajs = normalize(trajs)
        grad = policy_gradient(trajs, policy=agent)
        agentOpt.apply_gradient(grad)

        trainTimeLeft -= 1

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        world = Gym("BipedalWalker-v2", max_steps=MAX_STEPS)
        agent = walker()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        run()
