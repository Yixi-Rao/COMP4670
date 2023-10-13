import numpy as np
from scipy.stats import norm

# Functional Structure


def probability_improvement(X: np.ndarray, X_sample: np.ndarray,
                            gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Probability improvement acquisition function.

    Computes the PI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        PI: ndarray of shape (m,)
    """
    # TODO Q2.4
    # Implement the probability of improvement acquisition function
    f_plus    = max([gpr.predict(x) for x in X_sample])
    u_x, sd_x = gpr.predict(X, True)
    Z         = ((u_x - f_plus - xi) / sd_x).flatten()
    return norm.cdf(Z)


def expected_improvement(X: np.ndarray, X_sample: np.ndarray,
                         gpr: object, xi: float = 0.01) -> np.ndarray:
    """
    Expected improvement acquisition function.

    Computes the EI at points X based on existing samples X_sample using
    a Gaussian process surrogate model

    Arguments:
    ----------
        X: ndarray of shape (m, d)
            The point for which the expected improvement needs to be computed.

        X_sample: ndarray of shape (n, d)
            Sample locations

        gpr: GPRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.

        xi: float. Default 0.01
            Exploitation-exploration trade-off parameter.

    Returns:
    --------
        EI : ndarray of shape (m,)
    """

    # TODO Q2.4
    # Implement the expected improvement acquisition function

    f_plus    = max([gpr.predict(x) for x in X_sample]).flatten()
    u_x, sd_x = gpr.predict(X, True)
    u_x       = u_x.flatten()
    sd_x      = sd_x.flatten()
    Z         = ((u_x - f_plus - xi) / sd_x) if sd_x.all() > 0 else 0
    return (u_x - f_plus - xi) * norm.cdf(Z) + sd_x * norm.pdf(Z) if sd_x.all() > 0 else 0
