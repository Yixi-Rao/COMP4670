import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize

import numpy as npx
import boframework.gp as gp
import boframework.kernels as kernels

theta  = np.array([0.95006168,
                   -0.28380026])
kernel = kernels.Matern(
    length_scale=2.5858691463324135, variance=0.7529170156780511, nu=2.5)
gpr = gp.GPRegressor(kernel=kernel, noise_level=0.1)
gpr._kernel = kernel
print(tuple(kernel.bounds)[0])
gpr._X_train = np.array([[-0.5], [2.2]])
gpr._y_train = np.array([[-1.31584035], [-0.09311054]])
x, y = gpr.optimisation(gpr.negative_log_marginal_likelihood, gpr.kernel.theta, tuple(gpr.kernel.bounds))