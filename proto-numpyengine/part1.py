#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from bo import GaussianProcessOptimizer

np.random.seed(3049023405)

_N_GAUSSIANS = 7

_MU = [(np.random.rand(),np.random.rand()) for _ in range(_N_GAUSSIANS)]
_SIGMA = [np.random.rand()*0.12 + 0.04 for _ in range(_N_GAUSSIANS)]

_X_COEFFICIENT = np.random.rand() * 10
_Y_COEFFICIENT = np.random.rand() * 10

_NOISE_FACTOR = 0.01

_N = 120

#_MODE = 'normal'
_MODE = 'interactive'

def distance(p1, p2):
    """
    Compute distance between two points
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def f(x, y):
    """
    True function f()
    """
    gs = sum([norm.pdf(distance(_MU[i], (x,y)), scale=_SIGMA[i])
              for i in range(_N_GAUSSIANS)])

    return x*_X_COEFFICIENT + y*_Y_COEFFICIENT + gs


def sample(p):
    """
    Implement noisy sampler for f(x,y)
    """
    ares = f(p[0], p[1])
    return ares + (np.random.normal(loc=0, scale=_NOISE_FACTOR*ares))


if __name__ == '__main__':
    x_range = np.arange(0,1,0.01)
    y_range = np.arange(0,1,0.01)

    X,Y = np.meshgrid(x_range, y_range)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    result = []

    o = GaussianProcessOptimizer(sample, seed=np.random.randint(2560000))

    plt.ion()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Bayesian Optimization")
    plt.show()

    if _MODE == 'interactive':
        for i in range(_N):
            o.maximize(1)

            @np.vectorize
            def efunc(x, y):
                return o.predict((x,y))[0]

            @np.vectorize
            def vfunc(x, y):
                return o._acq((x,y))

            ax1.imshow(f(X,Y), cmap='hot', interpolation='nearest', vmin=0, vmax=20, extent=[0,1,1,0])
            ax2.imshow(efunc(X,Y), cmap='hot', interpolation='nearest', vmin=0, vmax=20, extent=[0,1,1,0])
            ax3.imshow(vfunc(X,Y), cmap='cool', interpolation='nearest', extent=[0,1,1,0])
            ax4.scatter(
                [k[0] for k in o.points.keys()],
                [k[1] for k in o.points.keys()],
                s=4, c='black',)
     
            ax1.set_title("True function $f(x, y)$") 
            ax2.set_title("Mean, estimated via Gaussian Process")
            ax3.set_title("Acquisition Function")
            ax4.set_title("Points Sampled")

            ax1.axis([0,1,0,1])
            ax2.axis([0,1,0,1])
            ax3.axis([0,1,0,1])
            ax4.axis([0,1,0,1])

            for g in [ax1,ax2,ax3,ax4]:
                g.axvline(o.last_point[0])
                g.axhline(o.last_point[1])
            
            ax4.axes.set_aspect('equal', None)

            plt.draw()
            plt.savefig('out/frame{:03d}.png'.format(i))
            plt.pause(0.01)

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
    else:
        o.maximize(_N)

        @np.vectorize
        def efunc(x, y):
            return o.predict((x,y))[0]

        @np.vectorize
        def vfunc(x, y):
            return o._acq((x,y))

        ax1.imshow(f(X,Y), cmap='hot', interpolation='nearest', vmin=0, vmax=20, extent=[0,1,1,0])
        ax2.imshow(efunc(X,Y), cmap='hot', interpolation='nearest', vmin=0, vmax=20, extent=[0,1,1,0])
        ax3.imshow(vfunc(X,Y), cmap='cool', interpolation='nearest', extent=[0,1,1,0])
        ax4.scatter(
            [k[0] for k in o.points.keys()],
            [k[1] for k in o.points.keys()],
            s=4, c='black',)
 
        ax1.set_title("True function $f(x, y)$") 
        ax2.set_title("Mean, estimated via Gaussian Process")
        ax3.set_title("Acquisition Function")
        ax4.set_title("Points Sampled")

        ax1.axis([0,1,0,1])
        ax2.axis([0,1,0,1])
        ax3.axis([0,1,0,1])
        ax4.axis([0,1,0,1]) 
        ax4.axes.set_aspect('equal', None)

        plt.draw()
        #plt.savefig('out/frame{:03d}.png'.format(i))
        plt.pause(100)


