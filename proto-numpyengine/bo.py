"""
Bayesian Optimization
Based on examples from: github.com/fmfn/BayesianOptimization

The optimizer at the page above is significantly more complex
than we require. We are able to cut down on the code size by
making assumptions about the Utility function and the dimensions
of the optimization. We have used some of the code, omitted portions,
and rewritten some portions for our own purposes.

Since this class would be considered a derivative work under copyright law,
we include the full original license text:

The MIT License (MIT)

Copyright (c) 2014 Fernando M. F. Nogueira

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

import scipy
import scipy.optimize

from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

_R = np.random.rand


class GaussianProcessOptimizer(object):
    def __init__(self, sample_function, kappa=20, seed=None, init_points=5, xi=0.01, **gp_params):
        self.f = sample_function

        # Seed the random number generator if seed was provided
        rand = np.random.RandomState() if seed is None else np.random.RandomState(seed)

        # GP Regression
        self.gp = GaussianProcessRegressor(
            #kernel=Matern(nu=2.5),
            kernel=RationalQuadratic(),
            n_restarts_optimizer=25,
            random_state=rand
        )

        self.gp.set_params(**gp_params)

        # Utility function parameter
        self.kappa = kappa

        self.xi = xi

        self._acq = lambda p: - self._expected_improvement(p)
        #self._acq = lambda p: - self._upper_confidence_bound(p)

        self.v_max = None

        # initial points, form { p : f(p) }
        self.points = {
                p : self.f(p) for p in [(_R(), _R()) for _ in range(init_points)]
        }

        self.v_max = max(self.points.values())
        
        self.update_fit()
        self.last_point = None

    def _upper_confidence_bound(self, p):
        mu, sigma = self.gp.predict(np.array([np.asarray(p)]), return_std=True)
        return mu + (self.kappa * sigma)

    def _expected_improvement(self, p):
        mu, sigma = self.gp.predict(np.array([np.asarray(p)]), return_std=True)
        z = (mu - self.v_max - self.xi)/sigma
        return (mu - self.v_max - self.xi) * norm.cdf(z) + sigma * norm.pdf(z)

    def update_fit(self):
        keys = [k for k in self.points.keys()]
        X = np.array([np.asarray(k) for k in keys])
        y = np.array([self.points[k] for k in keys])
        print(X, y)
        self.gp.fit(X, y)

    def _umax(self, i_points=5):
        ores = scipy.optimize.minimize(
            self._acq,
            [_R(), _R()],
            bounds=[(0,1), (0,1)])
        print(ores)
        return (ores.x[0], ores.x[1])

    def predict(self, p):
        r = self.gp.predict(np.array([np.asarray(p)]), return_std=True)
        return (r[0][0], r[1][0])

    def maximize(self, N=50):
        for _ in range(N):
            p = self._umax()
            i = 0
            while p in self.points:
                p = (_R(), _R())
                i += 1
                if i > 5:
                    print("Something nefarious is afoot.")

            v = self.f(p)
            if self.v_max is None or v > self.v_max:
                self.v_max = v
            self.points[p] = v
            self.update_fit()
            self.last_point = p


