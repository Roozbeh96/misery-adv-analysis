"""Turbulence statistics from ADV time series."""

import numpy as np


def reynolds_shear_stress(u, w, rotate=True):
    """Kinematic Reynolds shear stress -<u'w'> [m^2/s^2] at one sampling point.

    u : streamwise velocity time series [m/s] (flow direction positive)
    w : vertical velocity time series [m/s]

    With rotate=True, the (u, w) axes are first rotated by theta = atan(W/U) so that the
    mean vertical velocity is zero; this removes the effect of a small probe tilt, which
    otherwise mixes the normal stresses u'u' and w'w' into u'w'.

    Returns -<u'w'> and the rotation angle theta [deg].
    """
    ok = ~np.isnan(u) & ~np.isnan(w)
    u, w = np.asarray(u, float)[ok], np.asarray(w, float)[ok]
    theta = np.arctan2(w.mean(), u.mean()) if rotate else 0.0
    ur = u * np.cos(theta) + w * np.sin(theta)
    wr = -u * np.sin(theta) + w * np.cos(theta)
    return -np.mean((ur - ur.mean()) * (wr - wr.mean())), np.degrees(theta)
