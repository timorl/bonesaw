#!/usr/bin/env python3

import sys
import gym
import numpy as np
from matplotlib import pyplot as plt

sys.path.append("..")
from mannequin import Adam
from mannequin.autograd import AutogradLayer
from mannequin.basicnet import Affine, LReLU, Tanh, Input, Layer
from mannequin.logprob import Gauss

class DKLUninormal(AutogradLayer):
    def __init__(self, mean_logstd):
        import autograd.numpy as np

        const = mean_logstd.n_outputs//2

        def split(mls):
            space_dim = mls.shape[-1]//2
            return mls.T[:space_dim].T, np.clip(mls.T[space_dim:].T, -3., 3.)

        def dkl(mls):
            mean, logstd = split(mls)
            return 0.5 * ( np.sum(
                np.exp(logstd) - logstd + np.square(mean),
                axis=-1,
                keepdims=True) - const )

        super().__init__(mean_logstd, f=dkl, n_outputs=1)

class GaussDiagReparametrizationTrick(Layer):
    def __init__(self, mean_logstd):
        import autograd.numpy as np

        rng = np.random.RandomState()
        space_dim = mean_logstd.n_outputs//2

        def split(mls):
            return mls.T[:space_dim].T, np.clip(mls.T[space_dim:].T, -3., 3.)

        def evaluate(inps):
            mls, inner_backprop = mean_logstd.evaluate(inps)
            mean, logstd = split(mls)
            sample = rng.randn(space_dim)
            def backprop(grad):
                nonlocal logstd
                logstd = np.reshape(logstd, (-1, space_dim))
                grad = np.reshape(grad, (-1, space_dim))
                return inner_backprop(np.concatenate((grad, grad * sample * np.exp(logstd)), axis=-1))

            return mean + sample * np.exp(logstd), backprop

        super().__init__(mean_logstd, evaluate=evaluate, n_outputs=space_dim)

def run():
    mnist = np.load("__mnist.npz")
    mnist_train_x = mnist['train_x']

    encoder = Affine(Input(28*28), 300)
    encoder = Affine(Tanh(encoder), 2*5)
    dkl = DKLUninormal(encoder)
    encoded = GaussDiagReparametrizationTrick(encoder)

    latent = Input(5)
    decoder = Affine(latent, 300)
    decoder = Affine(Tanh(decoder), 28*28)

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    for i in range(10000):
        idx = np.random.choice(len(mnist_train_x), size=128)
        pics = mnist_train_x[idx]

        encoder.load_params(encOptimizer.get_value())
        decoder.load_params(decOptimizer.get_value())

        representation, encBackprop = encoded.evaluate(pics)

        changedPics, bckprp = decoder.evaluate(representation)
        decOptimizer.apply_gradient(bckprp(pics - changedPics))
        latentGrad = latent.last_gradient

        dklHD, bckprp = dkl.evaluate(pics)
        dklGrad = bckprp(-1.)

        reparametrizationTrickGrad = encBackprop(latentGrad)
        encOptimizer.apply_gradient(dklGrad + reparametrizationTrickGrad)

        print("Dec error: %f, DKL:%f"%(np.mean(np.abs(pics - changedPics)), np.mean(dklHD)))

        if i % 100 == 99:
            fig, plots = plt.subplots(2)
            plots[0].imshow(changedPics[43].reshape(28,28), cmap="gray", vmin=0, vmax=1)
            plots[1].imshow(pics[43].reshape(28,28), cmap="gray", vmin=0, vmax=1)
            fig.savefig("MVRT%05d.png"%i, dpi=100)


if __name__ == '__main__':
    run()
