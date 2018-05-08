# Numpy Bayesian Optimization Engine

You will need

- numpy
- scipy
- matplotlib
- sklearn

Then start the program with `./part1.py`. Some parameters are available to be
tweaked within this file, and lower-level parameters of the underlying models
can be tweaked within `bo.py`.

In particular, the `_MODE` variable within `part1.py` controls whether the
plots display every iteration. The program is much faster at computing a final
result if `_MODE` is ___not___ set to 'interactive'.
