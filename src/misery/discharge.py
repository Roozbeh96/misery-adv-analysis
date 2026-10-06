"""Discharge from velocity verticals (velocity-area method, trapezoidal rule)."""

import numpy as np


def depth_averaged_velocity(z, u, h):
    """Depth-averaged velocity and unit discharge of one vertical.

    z : heights of the measured points above the bed [m]
    u : time-averaged velocity at those heights [m/s]
    h : water depth at the vertical [m]

    The profile is closed with u = 0 at the bed (no slip) and with the top measured
    velocity held constant up to the surface, then integrated with the trapezoidal rule.

    Returns ubar [m/s] and q = ubar * h [m^2/s].
    """
    order = np.argsort(z)
    z, u = np.asarray(z, float)[order], np.asarray(u, float)[order]
    z_full = np.r_[0.0, z, h]
    u_full = np.r_[0.0, u, u[-1]]
    q = np.trapezoid(u_full, z_full)
    return q / h, q


def section_discharge(y, q, y_left, y_right):
    """Discharge through a cross-section from the unit discharge of its verticals.

    y       : lateral positions of the verticals [m]
    q       : unit discharge of each vertical [m^2/s]
    y_left  : lateral position of the left water edge [m]  (q = 0 there)
    y_right : lateral position of the right water edge [m] (q = 0 there)

    q is assumed to vary linearly between verticals and to fall linearly to zero at the
    water edges. Returns Q [m^3/s] and the contribution of each segment, from left to right.
    """
    order = np.argsort(y)
    y_full = np.r_[y_left, np.asarray(y, float)[order], y_right]
    q_full = np.r_[0.0, np.asarray(q, float)[order], 0.0]
    segments = (q_full[:-1] + q_full[1:]) / 2 * np.diff(y_full)
    return segments.sum(), segments
