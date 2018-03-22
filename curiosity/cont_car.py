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
from mannequin.backprop import autograd
from mannequin.distrib import Gauss, Discrete

def DKLUninormal(*, mean, logstd):
    @autograd
    def dkl(mean, logstd):
        import autograd.numpy as np
        return 0.5 * (
            np.sum(
                np.exp(logstd) - logstd + np.square(mean),
                axis=-1
            ) - mean.shape[-1]
        )

    return Function(mean, logstd, f=dkl, shape=())

def build_vae(*shape):
    def build_hidden(layer):
        for _ in range(2):
            layer = Tanh(Affine(layer, 128))
        return layer

    LATENT_SIZE=3
    normalize = RunningNormalize(shape=shape, horizon=50)

    encoder = build_hidden(Input(*shape))

    encoder = Gauss(mean=Affine(encoder, LATENT_SIZE), logstd=Affine(encoder, LATENT_SIZE))
    dkl = DKLUninormal(mean=encoder.mean, logstd=encoder.logstd)

    decoder_input = Input(LATENT_SIZE)
    decoder = build_hidden(decoder_input)
    decoder = Gauss(mean=Affine(decoder, *shape), logstd=Affine(decoder, *shape))

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    class Result:
        def train(inps):
            inps = normalize(inps)
            encoder.load_params(encOptimizer.get_value())
            decoder.load_params(decOptimizer.get_value())

            representation, encBackprop = encoder.sample.evaluate(inps)

            picsLogprob, decBackprop = decoder.logprob.evaluate(
                representation,
                sample=inps
            )

            dklValue, dklBackprop = dkl.evaluate(inps)

            decOptimizer.apply_gradient(
                decBackprop(np.ones(len(inps)))
            )

            encOptimizer.apply_gradient(
                dklBackprop(-np.ones(len(inps)))
                + encBackprop(decoder_input.last_gradient)
            )

        def generate(n):
            decoder.load_params(decOptimizer.get_value())

            lats = np.random.randn(n, LATENT_SIZE)
            return decoder.mean(lats)*normalize.get_std() + normalize.get_mean()

        def DKL(inps):
            inps = normalize.apply(inps)
            encoder.load_params(encOptimizer.get_value())
            return dkl(inps)

    return Result

def build_agent(env):
    inp_shape = env.observation_space.low.shape
    model = Input(*inp_shape)
    model = Tanh(Affine(model, 64))
    model = Tanh(Affine(model, 64))
    if isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.low.size
        Distribution = lambda p: Gauss(mean=p)
    elif isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
        Distribution = lambda p: Discrete(logits=p)
    else:
        raise ValueError("Unsupported action space")
    model = Distribution(Affine(model, action_size))

    normalize = RunningNormalize(shape=inp_shape)

    opt = Adam(
        np.random.randn(model.n_params),
        lr=0.005,
        horizon=10,
    )
    def train_step(traj):
        model.load_params(opt.get_value())
        _, backprop = model.logprob.evaluate(normalize.apply(traj.o), sample=traj.a)
        opt.apply_gradient(backprop(traj.r))

    def save(file_name):
        np.savez(file_name, params=model.get_params(), norm_mean=normalize.get_mean(), norm_std=normalize.get_std())

    def load(file_name):
        nonlocal normalize
        pars = np.load(file_name)
        model.load_params(pars["params"])
        mean = pars["norm_mean"]
        stdev = pars["norm_std"]
        def normalize(obs):
            return obs*stdev + mean


    model.policy = lambda o: model.sample(normalize(o))
    model.train_step = train_step
    model.save = save
    model.load = load
    return model

def save_plot(file_name, trajs, *,
        xs=lambda t: t.o[:,0],
        ys=lambda t: t.o[:,2],
        color=lambda t: ["b", "r"][np.argmax(t.a[0])]):
    import matplotlib.pyplot as plt
    plt.clf()
    for t in trajs:
        plt.plot(
            xs(t), ys(t),
            color=color(t), alpha=0.2, linewidth=2, zorder=1
        )
    plt.gcf().axes[0].set_ylim([-36., 36.])
    plt.gcf().axes[0].set_xlim([-88, 88])
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def traj_scorer(env):
    GEN_SEGM_LEN = 50
    imagination = build_vae(GEN_SEGM_LEN, *env.observation_space.low.shape)
    old_agent_trajs = []
    def get_segment_at(obs, start):
        return obs[start:start+GEN_SEGM_LEN]
    def split_into_segments(obs):
        assert(len(obs) >= GEN_SEGM_LEN)
        result = []
        for start in range(len(obs) - GEN_SEGM_LEN + 1):
            result.append(get_segment_at(obs, start))
        return np.array(result)
    def pick_segment(obs):
        start = np.random.randint(len(obs) - GEN_SEGM_LEN + 1)
        return get_segment_at(obs, start)
    def imagine(how_many):
        generated_obs = imagination.generate(how_many)
        return [Trajectory(go, [[1,0]]*GEN_SEGM_LEN) for go in generated_obs]
    def frolic():
        nonlocal old_agent_trajs
        old_agent_trajs = old_agent_trajs[-128:]
        agent_obs = np.array([pick_segment(at.o) for at in old_agent_trajs])
        imagination.train(agent_obs)
    def curious(obs):
        segmented_obs = split_into_segments(obs)
        rewards = imagination.DKL(segmented_obs)
        result = np.zeros(len(obs)) + np.mean(rewards)
        result[:len(rewards)] = rewards
        return result

    class Result:
        def score(agent_traj):
            old_agent_trajs.append(agent_traj)
            for _ in range(16):
                frolic()
            return agent_traj.modified(rewards=curious(agent_traj.o))
        def plot(file_name, transform=lambda x: x):
            generated_trajs = imagine(50)
            tagged_trajs = [Trajectory(agent_traj.o, [[0,1]]*len(agent_traj)) for agent_traj in old_agent_trajs[-20:]]
            forplot = generated_trajs + tagged_trajs
            forplot = [fp.modified(observations=transform) for fp in forplot]
            save_plot(file_name, forplot)
    return Result

def trainer(agent, world, scorer):
    rewardNormalize = RunningNormalize(horizon=10)

    def train(trainAgent=True):
        agent_traj = episode(world, agent.policy, max_steps=200)

        agent_traj_curio = scorer.score(agent_traj)

        print(bar(np.mean(agent_traj_curio.r), 200.))

        if trainAgent:
            agent_traj_curio = agent_traj_curio.discounted(horizon=200)
            agent_traj_curio = agent_traj_curio.modified(rewards=rewardNormalize)
            agent.train_step(agent_traj_curio)

    return train

def curiosity(world):
    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    agent = build_agent(world)
    scorer = traj_scorer(world)
    train = trainer(agent, world, scorer)

    def unnormalize(obs):
        return obs

    for ep in range(2000):
        train(ep>300)

        if (ep % 20) == 0:
            scorer.plot(log_dir + "/%04d.png"%ep, unnormalize)
            agent.save(log_dir + "/%04d.npz"%ep)


def render(world, agent_files):
    old_render = world.render
    def betterRender():
        import time
        old_render()
        time.sleep(0.01)
    world.render = betterRender
    agent = build_agent(world)
    for fn in agent_files:
        agent.load(fn)
        episode(world, agent.policy, render=True)

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
    world = make_env()

    if len(sys.argv) >= 2:
        render(world, sys.argv[1:])
    else:
        curiosity(world)

if __name__ == "__main__":
    run()
