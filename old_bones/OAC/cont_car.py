#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.append("../..")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
if "DEBUG" in os.environ:
    import IPython.core.ultratb
    sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

from worlds import Gym, ActionNoise, Cache
from models import Input, Affine, Softmax, LReLU
from trajectories import (normalize, discount, policy_gradient,
    print_reward, get_rewards, replace_rewards, retrace)
from optimizers import Adam, Adams

def build_agent():
    model = Input(2)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 1)
    return model

def build_oracle():
    model = Input(6)
    model = LReLU(Affine(model, 32))
    model = LReLU(Affine(model, 32))
    model = Affine(model, 2)
    return model

def save_plot(file_name, trajs, predictions):
    import matplotlib.pyplot as plt
    from matplotlib import collections as coll
    lines = []
    for i, (t, t_b) in enumerate(zip(trajs, predictions)):
        pos = np.array([0.25 * i, 0.0])
        pos_b = np.array([1.5 + 0.25 * i, 0.0])
        for (o1, a1, r1), (o2, a2, r2), delta_b in zip(t, t[10:], t_b):
            lines.append([pos, pos + (o2-o1)*[1.0, 10.0]])
            lines.append([pos_b, pos_b + delta_b*[1.0, 10.0]])
            pos = pos + [0.0, 0.01]
            pos_b = pos_b + [0.0, 0.01]
    lc = coll.LineCollection(lines, linewidths=1)
    plt.clf()
    plt.grid()
    plt.gcf().axes[0].set_ylim([-0.3, 5.3])
    plt.gcf().axes[0].set_xlim([-0.4, 3.0])
    plt.gcf().axes[0].add_collection(lc)
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def combine_rewards(trajss, weights):
    assert len(trajss) == len(weights)
    rewards = []
    for trajs in trajss:
        rewards.append(np.asarray([np.asarray([r for (_, _, r) in traj]) for traj in trajs]))
    rewards = [rew*weight for rew, weight in zip(rewards, weights)]
    rewards = np.sum(rewards, axis=0)
    return [[(o, a, r) for (o, a, _), r in zip(traj, rew)] for traj, rew in zip(trajss[0], rewards)]

def curiosity(world):
    world = ActionNoise(world, stddev=0.2)
    memory = Cache(max_size=100)

    log_dir = "__oracle"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent()
    agent_opt = Adams(
        np.random.randn(agent.n_params),
        lr=0.00015,
        memory=0.5
    )

    oracle = build_oracle()
    oracle_opt = Adam(
        np.random.randn(oracle.n_params) * 0.1,
        lr=0.05,
        memory=0.95
    )

    for episode in range(1000):
        agent.load_params(agent_opt.get_value())
        oracle.load_params(oracle_opt.get_value())

        agent_trajs = world.trajectories(agent, 4)
        for_oracle = [
            [
                (np.asarray([o1,o2, o3]).flatten(), a1, r1)
                for (o1, a1, r1), (o2, a2, r2), (o3, a3, r3)
                in zip(t, t[1:], t[2:])
            ]
            for t in agent_trajs
        ]
        memory.add_trajectories(for_oracle)

        predictions = retrace(for_oracle, model=oracle)
        save_plot(
            log_dir + "/%04d.png" % (episode + 1),
            agent_trajs, predictions
        )
        np.save(
            log_dir + "/%04d.npy" % (episode + 1),
            agent_opt.get_value()
        )

        curiosity_trajs = [
            [
                (o1, a1, np.log(np.mean(np.square((o2-o1) - delta_p))))
                for (o1, a1, r1), (o2, a2, r2), delta_p
                in zip(t, t[10:], p)
            ]
            for t, p in zip(agent_trajs, predictions)
        ]
        #curiosity_trajs = replace_rewards(curiosity_trajs,
        #    episode=lambda rs: np.max(rs))
        print_reward(curiosity_trajs, max_value=5000.0)
        print_reward(agent_trajs, max_value=90.0, episode=np.sum)

        curiosity_trajs = discount(curiosity_trajs, horizon=500)
        curiosity_trajs = normalize(curiosity_trajs)
        agent_trajs = discount(agent_trajs, horizon=500)
        agent_trajs = normalize(agent_trajs)
        agent_trajs = [traj[:-10] for traj in agent_trajs]
        agent_weight = 0.5# + 0.4*(0.5 * (1 - np.cos(np.pi * episode / 20)))
        curiosity_weight = 1. - agent_weight
        comb_trajs = combine_rewards([curiosity_trajs, agent_trajs], [curiosity_weight, agent_weight])
        grad = policy_gradient(comb_trajs, policy=agent)
        agent_opt.apply_gradient(grad)

        oracle_trajs = [
            [
                (o1, (o2-o1)[:2], 1.0)
                for (o1, a1, r1), (o2, a2, r2)
                in zip(t, t[10:])
            ]
            for t in memory.trajectories(None, 4)
        ]

        grad = policy_gradient(oracle_trajs, policy=oracle)
        oracle_opt.apply_gradient(grad)

def run():
    world = Gym("MountainCarContinuous-v0", max_steps=500)

    if len(sys.argv) >= 2:
        agent = build_agent()
        for fn in sys.argv[1:]:
            agent.load_params(np.load(fn))
            world.render(agent)
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
