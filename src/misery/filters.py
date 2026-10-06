"""ADV filtering. Python port of mPST_ADVSpikeFilter.m (ADV despiking, Parsheh et al. 2010 / Wahl 2003).

The port reproduces the MATLAB code line for line, including two quirks of the
original that affect the output:

* The "sample and hold" replacement uses ``find(idx < j, 1, 'last')``, which is a
  position *within* the list of valid indices, not a sample index. The replaced
  value is therefore ``U[count_of_valid_points_before_j]`` rather than the last
  valid sample. This is kept on purpose so that results match MATLAB.
* The validity flags ``n1`` (Step I) persist across the 6 passes.

The MATLAB inner ``for j = 2:num_samples`` loops are vectorised: the decision for
sample j only depends on values at the start of each pass, so only the replacement
itself (which can chain through earlier replaced samples) is done sequentially.
"""

import numpy as np
import matplotlib.pyplot as plt


def _matlab_mode(x):
    """MATLAB ``mode``: most frequent value, smallest one on ties, NaNs ignored."""
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    values, counts = np.unique(x, return_counts=True)
    return values[np.argmax(counts)]


def _matlab_median(x):
    """MATLAB ``median`` (returns NaN if any NaN is present)."""
    return np.median(x)


def _first_derivative(U2):
    """Finite-difference derivative exactly as written in mPST_ADVSpikeFilter.m."""
    n = len(U2)
    d = np.empty(n)
    d[: n - 2] = (4 * U2[1:-1] - U2[2:] - 3 * U2[:-2]) / 2
    d[n - 1] = U2[n - 1] - U2[n - 2]
    d[n - 2] = U2[n - 1] - U2[n - 2]
    return d


def _hold_replace(U, replace, valid_after):
    """Sequential sample-and-hold replacement of MATLAB loops (in place).

    ``replace``     : samples j (0-based, j >= 1) to replace in this pass
    ``valid_after`` : validity flags after this pass (only entries < j are used)
    """
    # number of valid samples strictly before each j
    count_before = np.concatenate(([0], np.cumsum(valid_after)[:-1]))
    for j in np.flatnonzero(replace):
        k = count_before[j]
        U[j] = U[j - 1] if k == 0 else U[k - 1]


def _despike(U, figs):
    U = np.asarray(U, dtype=float).copy()
    U_In = U.copy()

    # constants needed
    # C1 = 1.5 and C2 = 1.35 values used by Parsheh et al. 2010
    # 1<C1<2 and 1.25<C2<1.45 based on Parsheh 2010
    # Wahl 2003, C1 = C2 = 1.483
    c1 = 1.8  # can change these if data removal is not satisfactory
    c2 = 1.483  # this value came from Parsheh, 2010 as suggested by Wahl, 2003
    f = 64  # sampling frequency
    epsilon = 1e-10
    num_samples = len(U)
    log_term = np.sqrt(2 * np.log(num_samples))

    # remove mean value from the original timeseries
    U_NoMean = U - np.mean(U)
    n1_valid = np.ones(num_samples, dtype=bool)  # n1 == 9 in MATLAB

    # ---- STEP I (Identify Spiked Data Points) ----
    thetaU = _matlab_median(np.abs(U - _matlab_median(U)))
    Goodpoint = (U_NoMean <= c1 * thetaU) & (U_NoMean >= -c1 * thetaU)

    for _ in range(6):  # loop through 6 times checking for bad data points
        U_NoMean = U_NoMean - np.mean(U_NoMean)
        MAD = _matlab_median(np.abs(U_NoMean - _matlab_median(U_NoMean)))
        max_D = c2 * MAD * log_term
        exceed = np.abs(U_NoMean) >= max_D
        exceed[0] = False  # MATLAB loop starts at j = 2
        n1_valid &= ~exceed
        _hold_replace(U_NoMean, exceed, n1_valid)

    # ---- STEP II (Derivatives, Correction and Replacement, Ellipsoid) ----
    U2 = U_NoMean - np.mean(U_NoMean)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(6):
            Uprime = _first_derivative(U2)
            Uprime2 = _first_derivative(Uprime)

            # Thresholds
            Mad = _matlab_median(np.abs(U2 - _matlab_median(U2)))
            max_U = c2 * Mad * log_term
            Mad2 = _matlab_median(np.abs(Uprime - _matlab_median(Uprime)))
            max_deltaU = c2 * Mad2 * log_term
            c = max_deltaU + epsilon
            Mad3 = _matlab_median(np.abs(Uprime2 - _matlab_median(Uprime2)))
            max_delta2U = c2 * Mad3 * log_term

            # Correction and Replacement
            alpha = np.arctan(np.sum(U2 * Uprime2) / np.sum(U2**2))
            beta = np.arctan(Uprime / (U2 + epsilon))
            phi = np.arctan(U2 / (Uprime2 + epsilon) / (np.cos(beta) + epsilon))
            rho_m = Uprime2 / (np.cos(phi) + epsilon)
            a2 = 0.5 * (max_U**2 * (1 + 1 / np.cos(2 * alpha))) + 0.5 * (
                max_delta2U**2 * (1 - 1 / np.cos(2 * alpha))
            )
            b2 = max_delta2U**2 / np.cos(alpha) ** 2 - a2 * np.tan(alpha) ** 2
            inv = (
                (np.sin(phi) * np.cos(beta) * np.cos(alpha) + np.cos(phi) * np.sin(alpha)) ** 2 / a2
                + (np.sin(phi) * np.cos(beta) * np.sin(alpha) - np.cos(phi) * np.cos(alpha)) ** 2 / b2
                + (np.sin(phi) * np.sin(beta)) ** 2 / c**2
            )
            # MATLAB sqrt returns complex values for negative input; abs() is then the modulus
            rho = np.sqrt((1 / inv).astype(complex))

            # Make Ellipsoid
            replace = (np.abs(rho_m) > np.abs(rho)) & ~Goodpoint
            replace[0] = False  # MATLAB loop starts at j = 2
            Goodpoint = Goodpoint | replace
            _hold_replace(U2, replace, Goodpoint)

    new_U = U2 + np.mean(U_In)

    # this seems to correct nicely for heavily aliased data...
    temp_ADV_U = new_U - U_In  # offsets between filtered and raw
    off_U = _matlab_mode(temp_ADV_U)  # mode of those offset values
    U_new1 = new_U - off_U  # subtract offset from filtered

    if figs == 1:
        t = np.arange(1, num_samples + 1) / f
        plt.figure()
        plt.plot(t, U_In, "r-", label="Contaminated")
        plt.plot(t, U_new1, "b-", label="Clean")
        plt.title("Comparing the contaminated timeseries vs. the clean timeseries")
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (m/sec)")
        plt.legend()

        plt.figure()
        plt.plot(t, U_In, "r", label="raw")
        plt.plot(t, U_new1, "g", label="despiked and shifted")
        plt.plot(t, new_U, "b", label="despiked")
        plt.legend()
        plt.title("After alias correction")
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (m/sec)")

    return U_new1


def mPST_ADVSpikeFilter(U_raw, V_raw, W_raw, figs=0):
    """Despike the three ADV velocity components.

    Parameters
    ----------
    U_raw, V_raw, W_raw : array_like
        x-, y-, z-direction velocity components.
    figs : int
        0 (default) -> no plots, 1 -> plot raw vs. cleaned series.

    Returns
    -------
    UVW_new : ndarray, shape (N, 3)
        Corrected u, v, w velocity components (columns).
    """
    return np.column_stack([_despike(c, figs) for c in (U_raw, V_raw, W_raw)])
