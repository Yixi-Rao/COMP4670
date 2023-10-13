from __future__ import annotations

from boframework.kernels import Matern
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from typing import Callable, Tuple, Union, Type
import copy
from operator import itemgetter
import numpy as np

# Class Structure


class GPRegressor:
    """
    Gaussian process regression (GPR).

    Arguments:
    ----------
    kernel : kernel instance,
        The kernel specifying the covariance function of the GP.

    noise_level : float , default=1e-10
        Value added to the diagonal of the kernel matrix during fitting.
        It can be interpreted as the variance of additional Gaussian
        measurement noise on the training observations.

    n_restarts : int, default=0
        The number of restarts of the optimizer for finding the kernel's
        parameters which maximize the log-marginal likelihood. The first run
        of the optimizer is performed from the kernel's initial parameters,
        the remaining ones (if any) from thetas sampled log-uniform randomly
        (for more details: https://en.wikipedia.org/wiki/Reciprocal_distribution)
        from the space of allowed theta-values. If greater than 0, all bounds
        must be finite. Note that `n_restarts == 0` implies that one
        run is performed.

    random_state : RandomState instance
    """

    def __init__(self,
                 kernel: Matern,
                 noise_level: float = 1e-10,
                 n_restarts: int = 0,
                 random_state: Type[np.random.RandomState] = np.random.RandomState
                 ) -> None:

        self.kernel       = kernel
        self.noise_level  = noise_level
        self.n_restarts   = n_restarts
        self.random_state = random_state(4)

    def optimisation(self,
                     obj_func: Callable,
                     initial_theta: np.ndarray,
                     bounds: Tuple
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function that performs Quasi-Newton optimisation using L-BFGS-B algorithm.

        Note that we should frame the problem as a minimisation despite trying to
        maximise the log marginal likelihood.

        Arguments:
        ----------
        obj_func : the function to optimise as a callable
        initial_theta : the initial theta parameters, use under x0
        bounds : the bounds of the optimisation search

        Returns:
        --------
        theta_opt : the best solution x*
        func_min : the value at the best solution x, i.e, p*
        """
        # TODO Q2.2
        # Implement an L-BFGS-B optimisation algorithm using scipy.minimize built-in function
        result = minimize(obj_func, initial_theta, method="L-BFGS-B", bounds=bounds)
        return(result.x, result.fun)

    def fit(self, X: np.ndarray, y: np.ndarray) -> GPRegressor:
        """
        Fit Gaussian process regression model.

        Arguments:
        ----------
        X : ndarray of shape (n_samples, n_features)
            Feature vectors or other representations of training data.
        y : ndarray of shape (n_samples, n_targets)
            Target values.

        Returns:
        --------
        self : object
            The current GPRegressor class instance.
        """
        # TODO Q2.2
        '''
        # Fit the Gaussian process by performing hyper-parameter optimisation
        # using the log marginal likelihood solution. To maximise the marginal
        # likelihood, you should use the `optimisation` function

        # HINT I: You should run the optimisation (n_restarts) time for optimum results.

        # HINT II: We have given you a data structure for all hyper-parameters under the variable `theta`,
        #           coming from the Matern class. You can assume by optimising `theta` you are optimising
        #           all the hyper-parameters.

        # HINT III: Implementation detail - Note that theta contains the log-transformed hyperparameters
        #               of the kernel, so now we are operating on a log-space. So your sampling distribution
        #               should be uniform.
        '''
        self._kernel  = copy.deepcopy(self.kernel)
        self._X_train = X
        self._y_train = y
        curr_MLL      = float("inf")
        curr_theta    = self._kernel.theta
        
        for i in range(self.n_restarts):
            if i == 0:
                new_theta, MLL = self.optimisation(self.negative_log_marginal_likelihood, self._kernel.theta, self._kernel.bounds) # tuple(self._kernel.bounds)
            else:
                random_theta   = self.random_state.uniform(self._kernel.bounds[:, 0], self._kernel.bounds[:, 1])
                new_theta, MLL = self.optimisation(self.negative_log_marginal_likelihood, random_theta, self._kernel.bounds) # tuple(self._kernel.bounds)
            
            if MLL < curr_MLL:
                curr_MLL   = MLL
                curr_theta = new_theta
                
        self._kernel.theta = curr_theta
        self.kernel.theta  = curr_theta
        return self    

    def predict(self, X: np.ndarray, return_std: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using the Gaussian process regression model.

        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation (`return_std=True`).

        Arguments:
        ----------
        X : ndarray of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        Returns (depending on the case):
        --------------------------------
        y_mean : ndarray of shape (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.
        """
        # TODO Q2.2
        # Implement the predictive distribution of the Gaussian Process Regression
        # by using the Algorithm (1) from the assignment sheet.
        
        Ky     = self._kernel(self._X_train, self._X_train) + self.noise_level * np.eye(self._X_train.shape[0])
        L      = cholesky(Ky, lower=True)
        alpha  = cho_solve((L, True), self._y_train) 
        K_star = self._kernel(self._X_train, X)
        f_star = K_star.T @ alpha
        if return_std:
            V_star = []
            for i in range(X.shape[0]):
                x      = X[i]
                k_star = self._kernel(self._X_train, x)
                v      = solve_triangular(L, k_star, lower=True)
                v_star = self._kernel(x, x) - v.T @ v
                V_star.append(v_star.flatten())
            return (f_star, np.sqrt(np.array(V_star)))
        else:
            return f_star
        
    def fit_and_predict(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        return_std: bool = False, optimise_fit: bool = False
                        ) -> Union[
        Tuple[dict, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ]:
        """
        Predict and/or fit using the Gaussian process regression model.

        Based on the value of optimise_fit, we either perform predictions with or without fitting the GP first.

        Arguments:
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Feature vectors or other representations of training data.
        y_train : ndarray of shape (n_samples, n_targets)
            Target values.
        X_test : ndarray of shape (n_samples, n_features)
            Query points where the GP is evaluated.
        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.
        optimise_fit : bool, default=False
            If True, we first perform fitting and then we continue with the 
            prediction. Otherwise, perform only inference.

        Returns (depending on the case):
        --------------------------------
        kernel_params: the kernel (hyper)parameters, optional. Only for `optimise_fit=True` case; 
            HINT: use `get_params()` fuction from kernel object. 
        y_mean : ndarray of shape (n_samples, n_targets)
            Mean of predictive distribution a query points.
        y_std : ndarray of shape (n_samples, n_targets), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.

        """
        # TODO Q2.6a
        # Implement a fit and predict or only predict scenarios. The course of action
        # should be chosen based on the variable `optimise_fit`.
        if optimise_fit:
            gpr   = self.fit(X_train, y_train)
            y_all = gpr.predict(X_test, return_std)
            return (gpr.kernel.get_params(), y_all)
        else:
            self._kernel  = copy.deepcopy(self.kernel)
            self._X_train = X_train
            self._y_train = y_train
            y_all         = self.predict(X_test, return_std)
            return y_all

    def log_marginal_likelihood(self, theta: np.ndarray) -> float:
        """
        Return log-marginal likelihood of theta for training data.

        Arguments:
        ----------
        theta : ndarray of shape (n_kernel_params,)
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated.

        Returns:
        --------
        log_likelihood : float
            Log-marginal likelihood of theta for training data.
        """
        # TODO Q2.2
        # Compute the log marginal likelihood by using the Algorithm (1)
        # from the assignment sheet.
        kernel       = self._kernel
        kernel.theta = theta
        
        n     = self._y_train.shape[0]
        Ky    = kernel(self._X_train, self._X_train) + self.noise_level * np.eye(self._X_train.shape[0])
        L     = cholesky(Ky, lower=True)
        alpha = solve_triangular(L.T, solve_triangular(L, self._y_train, lower=True))
        
        return ((-1 / 2) * self._y_train.ravel().T @ alpha.ravel() - np.sum([np.log(L[i, i]) for i in range(L.shape[0])]) - (n / 2) * np.log(2 * np.pi))

    def negative_log_marginal_likelihood(self, theta: np.ndarray) -> float:
        """
        Return log-marginal likelihood of theta for training data.

        Arguments:
        ----------
        theta : ndarray of shape (n_kernel_params,)
            Kernel hyperparameters for which the log-marginal likelihood is
            evaluated.

        Returns:
        --------
        log_likelihood : float
            negative Log-marginal likelihood of theta for training data.
        """
        # Compute the negative log marginal likelihood by using the Algorithm (1)
        # from the assignment sheet.
        kernel       = self._kernel
        kernel.theta = theta
        
        n     = self._y_train.shape[0]
        Ky    = kernel(self._X_train, self._X_train) + self.noise_level * np.eye(self._X_train.shape[0])
        L     = cholesky(Ky, lower=True)
        alpha = solve_triangular(L.T, solve_triangular(L, self._y_train, lower=True))
        
        return -1 * ((-1 / 2) * self._y_train.ravel().T @ alpha.ravel() - np.sum([np.log(L[i, i]) for i in range(L.shape[0])]) - (n / 2) * np.log(2 * np.pi))