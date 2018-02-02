#!/usr/bin/env python3

import sys
import numpy as np
from mannequin.basicnet import Affine, Tanh, Input
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

def run():
    latent = Input(5)
    decoder = Affine(latent, 300)
    decoder = Affine(Tanh(decoder), 2*28*28)
    decoder.load_params(np.load(sys.argv[1]))

    def get_img(x):
        return decoder(x)[:28*28].reshape(28, 28)

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)

    img = ax.imshow(get_img(np.zeros(latent.n_outputs)), cmap="gray", vmin=0, vmax=1)

    slider_ax = [plt.axes([0.05, 0.05 * (i+1), 0.9, 0.03]) for i in range(latent.n_outputs)]
    sliders = [Slider(a, '', -2.0, 2.0, valinit=0.0) for a in slider_ax]

    def update(val):
        x = [s.val for s in sliders]
        img.set_data(get_img(x))
        fig.canvas.draw_idle()

    for s in sliders:
        s.on_changed(update)

    plt.show()

if __name__ == "__main__":
    run()
