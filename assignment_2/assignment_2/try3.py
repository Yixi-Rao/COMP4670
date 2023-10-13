import numpy as np
import boframework.gp as gp
import boframework.kernels as kernels
import boframework.acquisitions as acquisitions
import boframework.bayesopt as bayesopt
import unittest
from typing import Callable, Tuple
import numpy as np
from scipy.optimize import minimize
from boframework.kernels import Matern
from boframework.gp import GPRegressor
import matplotlib.pyplot as plt
from scipy.linalg import cho_solve, cholesky, solve_triangular

def f(X, noise_level=0.1):
    return np.sin(X) + np.sin(2 * X) + noise_level * np.random.randn(*X.shape)

X_init = np.array([[-0.5], [2.2]])
Y_init = f(X_init)

m52 = Matern(length_scale=1.0, variance=1.0, nu=2.5)
gpr = GPRegressor(kernel=m52, noise_level=0.1, n_restarts=5)
ori_theta = gpr.kernel.theta


gpr2            = gpr.fit(X_init, Y_init)
res_theta = gpr2._kernel.theta

# L = np.array([[3.04279986, 0.        ],
#               [0.,         3.04279986]])
# Y = np.array([[-1.31584035],[-0.09311054]])
# alpha = solve_triangular(L.T, solve_triangular(L, Y, True))

    



