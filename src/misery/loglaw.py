"""Fit of the logarithmic law of the wall to a mean velocity profile."""

import numpy as np

KAPPA = 0.39  # von Karman constant


def fit_log_law(z, u, kappa=KAPPA):
    """Least-squares fit of  u / u_tau = (1/kappa) ln(z / z0)  to one vertical.

    z : heights of the measured points above the bed [m]
    u : time-averaged streamwise velocity at those heights [m/s] (flow direction positive)

    The law is linear in ln z:  u = a ln z + b,  with  a = u_tau / kappa  and  b = -a ln z0,
    so u_tau = kappa * a and z0 = exp(-b / a).

    Returns u_tau [m/s], z0 [m] and R^2 of the fit.
    """
    z, u = np.asarray(z, float), np.asarray(u, float)
    a, b = np.polyfit(np.log(z), u, 1)
    u_fit = a * np.log(z) + b
    r2 = 1 - np.sum((u - u_fit) ** 2) / np.sum((u - u.mean()) ** 2)
    return kappa * a, np.exp(-b / a), r2
