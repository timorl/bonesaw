#!/usr/bin/env python3

import sys
import gym
import numpy as np
from matplotlib import pyplot as plt

#import IPython.core.ultratb
#sys.excepthook = IPython.core.ultratb.FormattedTB(call_pdb=True)

sys.path.append("..")
from mannequin import Adam, bar
from mannequin.backprop import autograd
from mannequin.distrib import Gauss
from mannequin.basicnet import Affine, LReLU, Tanh, Input, Function, Params, Const, Layer

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

def Split(inner):
    secondEvaluate = None
    class First(Layer):
        def __init__(self):

            def evaluate(inps):
                nonlocal secondEvaluate
                firstResult = None
                firstBackprop = None
                firstGrad = None
                def backprop(grad, output=None):
                    nonlocal firstGrad
                    assert(firstGrad is None)
                    firstGrad = grad
                    return []

                firstResult, firstBackprop = inner.evaluate(inps)
                def newEvaluate(inps):
                    nonlocal firstResult
                    def newBackprop(grad, output=None):
                        nonlocal firstGrad, firstBackprop
                        firstGrad += grad
                        result = firstBackprop(firstGrad, output=output)
                        firstBackprop = None
                        firstGrad = None
                        return result
                    res = firstResult
                    firstResult = None
                    return res, newBackprop
                assert(secondEvaluate is None)
                secondEvaluate = newEvaluate
                return firstResult, backprop

            super().__init__(evaluate=evaluate, shape=inner.shape, n_params=0)
    class Second(Layer):
        def __init__(self):

            def evaluate(inps):
                nonlocal secondEvaluate
                eva = secondEvaluate
                secondEvaluate = None
                return eva(inps)

            super().__init__(evaluate=evaluate, shape=inner.shape, n_params=inner.n_params, get_params=inner.get_params, load_params=inner.load_params)

    return First(), Second()

def Clip(inner, a, b):
   def f(v):
       multiplier = np.ones(v.shape)
       multiplier[v < a] = 0.0
       multiplier[v > b] = 0.0
       return np.clip(v, a, b), lambda g: (g * multiplier,)
   return Function(inner, f=f)

def Times(inner, a):
   def f(v):
       return v * a, lambda g: (g * a,)
   return Function(inner, f=f)

def run():
    train_x = np.load("__mnist.npz")['train_x']

    encoder = Input(28*28)
    encoder = Tanh(Affine(encoder, 300))
    #encoder = Split(encoder)
    encoder = Affine(encoder, 5), Params(5)

    dkl = DKLUninormal(mean=encoder[0], logstd=encoder[1])
    encoder = Gauss(mean=encoder[0], logstd=encoder[1])

    decoder_input = Input(5)
    decoder = Tanh(Affine(decoder_input, 300))
    decoder = Split(decoder)
    decoder = Affine(decoder[0], 28*28), Times(Affine(decoder[1], 28*28), 0.001)
    meanSav = decoder[0]
    logstdSav = decoder[1]
    decoder = Gauss(mean=decoder[0], logstd=decoder[1])

    encOptimizer = Adam(encoder.get_params(), horizon=10, lr=0.01)
    decOptimizer = Adam(decoder.get_params(), horizon=10, lr=0.01)

    for i in range(10000):
        idx = np.random.choice(len(train_x), size=128)
        pics = train_x[idx]*100

        encoder.load_params(encOptimizer.get_value())
        decoder.load_params(decOptimizer.get_value())

        representation, encBackprop = encoder.sample.evaluate(pics)

        picsLogprob, decBackprop = decoder.logprob.evaluate(
            representation,
            sample=pics
        )

        dklValue, dklBackprop = dkl.evaluate(pics)

        decOptimizer.apply_gradient(
            decBackprop(np.ones(128))
        )

        encOptimizer.apply_gradient(
            dklBackprop(-np.ones(128))
            + encBackprop(decoder_input.last_gradient)
        )

        print("Logprob:", bar(np.mean(picsLogprob), 20000),
            "DKL:", bar(np.mean(dklValue), 200))

        if i % 100 == 99:
            plt.clf()
            fig, plots = plt.subplots(4)
            changedPic = decoder.sample(representation[43])
            curMean, _ = meanSav.evaluate(representation[43])
            curLogstd, _ = logstdSav.evaluate(representation[43])
            meanLogstd = np.mean(curLogstd)
            print(meanLogstd)
            plots[0].imshow(changedPic.reshape(28,28),
                cmap="gray", vmin=0, vmax=100)
            plots[1].imshow(pics[43].reshape(28,28),
                cmap="gray", vmin=0, vmax=100)
            plots[2].imshow(curMean.reshape(28,28),
                cmap="gray", vmin=0, vmax=100)
            plots[3].imshow(np.exp(curLogstd).reshape(28,28),
                cmap="gray", vmin=0, vmax=10)
            fig.savefig("step_%05d.png"%(i+1), dpi=100)

        if i % 1000 == 999:
            np.save("step_%05d_decoder.npy"%(i+1), decoder.get_params())

if __name__ == '__main__':
    run()
