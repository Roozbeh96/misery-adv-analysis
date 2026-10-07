"""Plotting functions for the ADV velocity field (ported from main.m)."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misery.plotting_site import draw_water_edges

# LaTeX-like look without requiring a TeX installation (MATLAB 'Interpreter','latex')
plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

MATLAB_BLUE = (0, 0.4470, 0.7410)  # default first line colour in MATLAB


def matlab_quiver_autoscale(x, y, u, v, scale):
    """Replicates MATLAB quiver's automatic arrow scaling (vector inputs)."""
    n = m = np.sqrt(x.size)
    delx = (x.max() - x.min()) / n
    dely = (y.max() - y.min()) / m
    dels = delx**2 + dely**2
    maxlen = np.sqrt((u**2 + v**2) / dels).max() if dels > 0 else 0
    return scale * 0.9 / maxlen if maxlen > 0 else scale * 0.9


def matlab_quiver3_autoscale(x, y, z, u, v, w, scale):
    """Replicates MATLAB quiver3's automatic arrow scaling (vector inputs)."""
    n = m = np.sqrt(x.size)
    delx = (x.max() - x.min()) / n
    dely = (y.max() - y.min()) / m
    delz = (z.max() - z.min()) / max(m, n)
    dels = np.sqrt(delx**2 + dely**2 + delz**2)
    maxlen = np.sqrt((u / dels) ** 2 + (v / dels) ** 2 + (w / dels) ** 2).max() if dels > 0 else 0
    return scale * 0.9 / maxlen if maxlen > 0 else scale * 0.9


def plot_3d_field(ParentDataset, Y1, Y2, Y3):
    """3D sampling locations and mean velocity vectors (Figure 1 of main.m).

    Y1, Y2, Y3 : Y coordinates of the three stations along X1, X2, X3.
    """
    fields = list(ParentDataset)
    numDatasets = len(fields)

    locs = np.zeros((numDatasets, 3))  # [X, Y, Z]
    mean_UVW = np.zeros((numDatasets, 3))  # [Mean_u, Mean_v, Mean_w]
    mag_U = np.zeros(numDatasets)  # Magnitude of mean velocity

    # Loop through datasets and calculate averages
    for i, name in enumerate(fields):
        ds = ParentDataset[name]
        locs[i, :] = ds["location"]
        mU = np.nanmean(ds["u"])
        mV = np.nanmean(ds["v"])
        mW = np.nanmean(ds["w"])
        # mean_UVW[i, :] = [mU, mV, mW]
        mean_UVW[i, :] = [mU, 0, 0]
        mag_U[i] = np.linalg.norm([mU, mV, mW])

    # 3. Create 3D Plot
    fig = plt.figure(num="3D Mean Velocity Field", facecolor="w", figsize=(14, 9))
    ax = fig.add_subplot(projection="3d")

    # Grid dimensions
    X_coords = np.array([-1.75, -6.75, -11.5])
    Y_mesh = np.array([Y1, Y2, Y3]).T

    # Water depth for the 9 (X,Y) grid points (X1Y1-3, X2Y1-3, X3Y1-3)
    w_d = np.array([0.83, 0.72, 0.64, 0.68, 0.6, 0.49, 0.48, 0.44, 0.52])
    Z_b = np.array([57.02, 57.02, 57.02, 57.01, 56.89, 57.09, 57.25, 57.23, 57.29])

    Z_water_grid = (w_d + Z_b).reshape((3, 3), order="F")  # 3x3 matching (Y, X) grid
    X_mesh, _ = np.meshgrid(X_coords, np.arange(1, 4))

    # Faint free-water surface (top boundary)
    ax.plot_surface(
        X_mesh, Y_mesh, Z_water_grid,
        color=(0.30, 0.75, 0.93), alpha=0.25,
        edgecolor=(0.1, 0.4, 0.7), linewidth=1.0, shade=False,
    )
    # Bed / bottom surface
    ax.plot_surface(
        X_mesh, Y_mesh, Z_b.reshape((3, 3), order="F"),
        color=(1, 0.8, 0.8), alpha=0.25,
        edgecolor=(0.5, 0.5, 0.5), linestyle="--", linewidth=0.5, shade=False,
    )

    # 3D quiver (vectors proportional to mean u, v, w; 0.5 scale as in MATLAB)
    s = matlab_quiver3_autoscale(
        locs[:, 0], locs[:, 1], locs[:, 2], mean_UVW[:, 0], mean_UVW[:, 1], mean_UVW[:, 2], 0.5
    )
    ax.quiver(
        locs[:, 0], locs[:, 1], locs[:, 2],
        mean_UVW[:, 0] * s, mean_UVW[:, 1] * s, mean_UVW[:, 2] * s,
        color=MATLAB_BLUE, linewidth=1.5, arrow_length_ratio=0.2,
    )

    # Scatter overlay at sampling points, colour-coded by velocity magnitude
    sc = ax.scatter(
        locs[:, 0], locs[:, 1], locs[:, 2], s=80, c=mag_U,
        cmap="jet", vmin=mag_U.min(), vmax=mag_U.max(),
        edgecolors="k", depthshade=False,
    )

    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(
        r"$\bar{U}_{\mathrm{mag}} = \sqrt{\bar{u}^2 + \bar{v}^2 + \bar{w}^2}$", fontsize=12
    )
    ax.set_xlabel(r"$X\ \mathrm{[m]}$", fontsize=12)
    ax.set_ylabel(r"$Y\ \mathrm{[m]}$", fontsize=12)
    ax.set_zlabel(r"$Z\ \mathrm{[m]}$", fontsize=12)
    ax.set_title("3D Sampling Locations & Mean Velocity Vectors", fontsize=14, fontweight="bold")
    ax.tick_params(labelsize=12)

    ax.invert_xaxis()  # XDir reverse
    ax.invert_yaxis()  # YDir reverse
    ax.view_init(elev=30, azim=-37.5 - 90)  # MATLAB view(3): az = -37.5, el = 30
    ax.set_aspect("equal")  # axis equal

    # Label points with dataset names
    for i, name in enumerate(fields):
        ax.text(locs[i, 0], locs[i, 1], locs[i, 2], "  " + rf"$\mathtt{{{name}}}$", fontsize=12)

    return fig


def plot_planform(ParentDataset, water_edges=None):
    """Planform time-averaged velocity at the target elevations (Figure 2 of main.m).

    water_edges : optional (X, Y_left, Y_right) of the water edges on both banks.
    Also prints and returns the summary table.
    """
    # 1. Target grid matrix [xIdx, yIdx] -> zIdx
    # Row 1: (x1,y1,z6), (x1,y2,z6), (x1,y3,z3)
    # Row 2: (x2,y1,z5), (x2,y2,z5), (x2,y3,z3)
    # Row 3: (x3,y1,z4), (x3,y2,z3), (x3,y3,z3)
    gridMatrix = np.array([[6, 6, 3], [5, 5, 3], [4, 3, 3]])

    # 2. Extract data for target grid points
    numPoints = 9
    Prefix_List = [""] * numPoints
    X_locs = np.zeros(numPoints)
    Y_locs = np.zeros(numPoints)
    Z_locs = np.zeros(numPoints)
    U_timeAvg = np.zeros(numPoints)
    V_timeAvg = np.zeros(numPoints)
    W_timeAvg = np.zeros(numPoints)
    Mag_timeAvg = np.zeros(numPoints)

    idx = 0
    for xIdx in range(1, 4):
        for yIdx in range(1, 4):
            zIdx = gridMatrix[xIdx - 1, yIdx - 1]
            targetPrefix = f"x{xIdx}y{yIdx}z{zIdx}"
            Prefix_List[idx] = targetPrefix

            if targetPrefix in ParentDataset:
                ds = ParentDataset[targetPrefix]
                X_locs[idx], Y_locs[idx], Z_locs[idx] = ds["location"]
                U_timeAvg[idx] = np.nanmean(ds["u"])
                V_timeAvg[idx] = np.nanmean(ds["v"])
                W_timeAvg[idx] = np.nanmean(ds["w"])
                Mag_timeAvg[idx] = np.linalg.norm([U_timeAvg[idx], V_timeAvg[idx], W_timeAvg[idx]])
            else:
                warnings.warn(f"Prefix {targetPrefix} not found in ParentDataset.")
            idx += 1

    # 3. Summary table
    SummaryTable = pd.DataFrame(
        {
            "Prefix": Prefix_List, "X_m": X_locs, "Y_m": Y_locs, "Z_m": Z_locs,
            "U_bar": U_timeAvg, "V_bar": V_timeAvg, "W_bar": W_timeAvg, "Mag_bar": Mag_timeAvg,
        }
    )
    print("--- Time-Averaged Velocities at Specific Wall-Normal Locations ---")
    print(SummaryTable.to_string(index=False))

    # 4. Planform (2D) time-averaged velocity field
    fig, ax = plt.subplots(num="Time-Averaged Velocity Field at Selected Z", facecolor="w", figsize=(14, 9))
    fig.subplots_adjust(left=0.05, right=0.82)  # leave room on the right for the point labels
    ax.grid(True, alpha=0.3)
    if water_edges is not None:
        draw_water_edges(ax, *water_edges)

    # Quiver of planar vectors (U_bar, 0); MATLAB scale factor 0.5
    V_zero = np.zeros_like(U_timeAvg)
    s = matlab_quiver_autoscale(X_locs, Y_locs, U_timeAvg, V_zero, 0.5)
    ax.quiver(
        X_locs, Y_locs, U_timeAvg * s, V_zero * s,
        angles="xy", scale_units="xy", scale=1, color="k", width=0.002,
    )

    # Scatter overlay colour-coded by total 3D time-averaged magnitude
    sc = ax.scatter(X_locs, Y_locs, s=120, c=Mag_timeAvg, cmap="jet", edgecolors="k", zorder=3)

    # MATLAB axis limits include the arrow tips and snap to tick values
    ax.update_datalim(np.column_stack([X_locs + U_timeAvg * s, Y_locs + V_zero * s]))
    with plt.rc_context({"axes.autolimit_mode": "round_numbers"}):
        ax.autoscale_view()
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # Colorbar on the left of the axes so it does not cover the point labels on the right
    cb = fig.colorbar(sc, ax=ax, location="left", pad=0.1)
    cb.set_label(
        r"$\bar{U} = \sqrt{\bar{u}^2 + \bar{v}^2 + \bar{w}^2}\ \mathrm{[m/s]}$", fontsize=14
    )
    cb.ax.tick_params(labelsize=18)
    ax.set_xlabel(r"$X\ \mathrm{[m]}$", fontsize=14)
    ax.set_ylabel(r"$Y\ \mathrm{[m]}$", fontsize=14)
    ax.set_title(
        "Time-Averaged Velocity Field at Target Wall-Normal Elevations", fontsize=14, fontweight="bold"
    )
    ax.tick_params(labelsize=18)

    # Annotate each point with magnitude, z index (z_i) and Z elevation
    for p in range(numPoints):
        xIdx, yIdx = divmod(p, 3)  # MATLAB ind2sub([3,3], p) -> (yIdx, xIdx)
        zIdx = gridMatrix[xIdx, yIdx]
        u_bar = r"$\bar{u}$"  # LaTeX kept outside the f-string, so its braces are not read as Python
        labelStr = f" {u_bar} = {U_timeAvg[p]:.3f} m/s ($z_{{{zIdx}}} = {Z_locs[p]:.2f}\\,\\mathrm{{m}}$)"
        ax.text(X_locs[p], Y_locs[p], labelStr, fontsize=14, va="bottom")

    return fig, SummaryTable
