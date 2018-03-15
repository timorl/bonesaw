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
from mannequin.distrib import Gauss

GEN_SEGM_LEN = 25

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

def build_hidden(layer):
    for _ in range(2):
        layer = Tanh(Affine(layer, 128))
    return layer

def build_vae():
    LATENT_SIZE=2
    encoder = build_hidden(Input(2*GEN_SEGM_LEN))
    encoder = Affine(encoder, LATENT_SIZE), Params(LATENT_SIZE)

    dkl = DKLUninormal(mean=encoder[0], logstd=encoder[1])
    encoder = Gauss(mean=encoder[0], logstd=encoder[1])

    decoder_input = Input(LATENT_SIZE)
    decoder = build_hidden(decoder_input)
    decoder = Affine(decoder, 2*GEN_SEGM_LEN)
    mean_dec = decoder
    decoder = Gauss(mean=decoder)

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    class Result:
        def train(inps):
            assert(len(inps.shape) == 2)
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

            lats = np.random.randn(n, LATENT_SIZE)*2
            return mean_dec(lats)

    return Result

def split_obs(obs):
    cutoff = len(obs)%(2*GEN_SEGM_LEN)
    obs = obs[cutoff:]
    #obs = obs.reshape(GEN_SEGM_LEN, -1, 2).transpose((1,0,2))
    return obs.reshape(-1, 2*GEN_SEGM_LEN)

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

def save_plot(file_name, trajs, *,
        xs=lambda t: t.o[:,0],
        ys=lambda t: t.o[:,1],
        color=lambda t: ["b", "r"][np.argmax(t.a[0])]):
    import matplotlib.pyplot as plt
    plt.clf()
    for t in trajs:
        plt.plot(
            xs(t), ys(t),
            color=color(t), alpha=0.2, linewidth=2, zorder=1
        )
    plt.gcf().axes[0].set_ylim([-0.075, 0.075])
    plt.gcf().axes[0].set_xlim([-1.25, 0.5])
    plt.gcf().set_size_inches(10, 8)
    plt.gcf().savefig(file_name, dpi=100)

def traj_scorer():
    imagination = build_vae()
    classifier = build_classifier()
    old_agent_trajs = []
    forplot = []
    def imagine():
        generated_obs = [go.reshape(-1, 2) for go in imagination.generate(200//GEN_SEGM_LEN)]
        return [Trajectory(go, [[1,0]]*GEN_SEGM_LEN) for go in generated_obs]
    def frolic(agent_traj):
        nonlocal old_agent_trajs
        old_agent_trajs = old_agent_trajs[-128:]
        old_agent_trajs.append(agent_traj)
        agent_obs = np.concatenate([split_obs(at.o) for at in old_agent_trajs[-50:]], axis=0)
        imagination.train(agent_obs)
    def ponder(agent_traj, generated_trajs):
        nonlocal forplot
        tagged_traj = Trajectory(agent_traj.o, [[0,1]]*len(agent_traj))
        classifier_traj = Trajectory.joined(tagged_traj, *generated_trajs)
        classifier.sgd_step(classifier_traj, lr=0.001)

        forplot += [tagged_traj, *generated_trajs]
    class Result:
        def score(agent_traj):
            generated_trajs = imagine()
            frolic(agent_traj)
            ponder(agent_traj, generated_trajs)
            return agent_traj.modified(
                    rewards=lambda r: classifier.predict(agent_traj.o)[:,0]
            )
        def plot(file_name, transform=lambda x: x):
            nonlocal forplot
            forplot = [fp.modified(observations=transform) for fp in forplot]
            save_plot(file_name, forplot)
            forplot = []
    return Result

def curiosity(world):
    log_dir = "__car"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    world = NormalizedObservations(world)
    agent = build_agent()
    scorer = traj_scorer()

    curNormalize = RunningNormalize(horizon=10)

    def unnormalize(obs):
        return (obs * world.get_std()) + world.get_mean()

    for ep in range(2000):
        agent.randomize_policy()
        agent_traj = episode(world, agent.policy, max_steps=200)

        agent_traj_curio = scorer.score(agent_traj)

        if (ep % 20) == 0:
            scorer.plot(log_dir + "/%04d.png"%ep, unnormalize)

        print(bar(np.mean(agent_traj_curio.r), 1.0))

        agent_traj_curio = agent_traj_curio.discounted(horizon=200)
        agent_traj_curio = agent_traj_curio.modified(rewards=curNormalize)
        agent.sgd_step(agent_traj_curio)

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
