"""Site features shared by the ADV and ADCP plots."""

import numpy as np

WATER_COLOR = (0.30, 0.75, 0.93)
EDGE_COLOR = (0.1, 0.4, 0.7)


def draw_water_edges(ax, X, Y_left, Y_right, label=True):
    """Planform (X horizontal, Y vertical): water edges on both banks at the surveyed stations,
    connected by lines, with the wetted area shaded in between.

    X       : streamwise positions of the stations [m]
    Y_left  : Y of the water edge on the left bank at each station [m]
    Y_right : Y of the water edge on the right bank at each station [m]
    """
    order = np.argsort(X)
    X, Y_left, Y_right = (np.asarray(a, float)[order] for a in (X, Y_left, Y_right))

    ax.fill_between(X, Y_left, Y_right, color=WATER_COLOR, alpha=0.12, linewidth=0, zorder=0)
    for Y, name in ((Y_left, "left"), (Y_right, "right")):
        ax.plot(X, Y, "-o", color=EDGE_COLOR, linewidth=1.8, markersize=5, zorder=2,
                label="water edge" if (label and name == "left") else None)
    if label:
        x_mid = X.mean()
        ax.text(x_mid, np.interp(x_mid, X, Y_left) - 0.15, "left bank (water edge)", ha="center", va="top",
                fontsize=11, color=EDGE_COLOR, style="italic")
        ax.text(x_mid, np.interp(x_mid, X, Y_right) + 0.15, "right bank (water edge)", ha="center",
                va="bottom", fontsize=11, color=EDGE_COLOR, style="italic")
