"""Plotting functions for the ADCP (SonTek SL1500-3G) velocity dataset."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

# LaTeX-like look without requiring a TeX installation (same style as plotting_ADV)
plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.serif": ["STIXGeneral"],
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
    }
)

BEAM_ANGLE_DEG = 25.37  # SL1500 beam angle, from Profile_X_Vel = (b0 - b1) / (2 sin(theta))
STATION_COLORS = {"X1": "tab:blue", "X2": "tab:orange", "X3": "tab:green"}


def plot_U_profiles(ADCPDataset, ax=None, theta_deg=BEAM_ANGLE_DEG, water_level=None):
    """U across Y for each station, with a shaded +/- U error band (instrument error).

    water_level : water-surface elevation [m]; if given, the legend shows how far each
                  instrument is below the surface.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for station, d in ADCPDataset.groupby("Station"):
        Y = d["Y_m[m]"].to_numpy()
        U = d["U[m/s]"].to_numpy()
        U_err = d["Error[m/s]"].to_numpy() / (2 * np.sin(np.radians(theta_deg)))
        color = STATION_COLORS.get(station)
        Z = d["Z_m[m]"].iloc[0]
        label = rf"{station}  ($X = {d['X_m[m]'].iloc[0]:.2f}$ m, $Z = {Z:.2f}$ m"
        label += rf", {water_level - Z:.2f} m below surface)" if water_level is not None else ")"

        ax.fill_between(Y, U - U_err, U + U_err, color=color, alpha=0.2, linewidth=0)
        ax.plot(Y, U, "-o", color=color, markersize=3, linewidth=1.2, label=label)

    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel(r"$Y\ \mathrm{[m]}$", fontsize=14)
    ax.set_ylabel(r"$U\ \mathrm{[m/s]}$", fontsize=14)
    ax.set_title(r"Lateral profiles of $U$ (shaded: $\pm\sigma_U$)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=12)
    return ax


def plot_U_planform(ADCPDataset, ax=None, arrow_scale=1.0):
    """Planform map (X horizontal, Y vertical): cells coloured by U, arrows show (U, V).

    arrow_scale : arrow length [m] drawn for a velocity of 1 m/s.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    X = ADCPDataset["X_m[m]"].to_numpy()
    Y = ADCPDataset["Y_m[m]"].to_numpy()
    U = ADCPDataset["U[m/s]"].to_numpy()
    V = ADCPDataset["V[m/s]"].to_numpy()

    # Diverging colormap centred on U = 0, so reversed flow stands out
    U_lim = np.nanmax(np.abs(U))
    sc = ax.scatter(X, Y, c=U, cmap="RdBu_r", norm=TwoSlopeNorm(0, -U_lim, U_lim),
                    s=60, marker="s", edgecolors="k", linewidths=0.3, zorder=3)
    ax.quiver(X, Y, U * arrow_scale, V * arrow_scale, angles="xy", scale_units="xy", scale=1,
              width=0.002, color="k", zorder=4)

    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$U\ \mathrm{[m/s]}$", fontsize=14)
    for station, d in ADCPDataset.groupby("Station"):
        ax.text(d["X_m[m]"].iloc[0], d["Y_m[m]"].min() - 0.3, station,
                ha="center", va="top", fontsize=12, fontweight="bold")

    ax.set_xlabel(r"$X\ \mathrm{[m]}$", fontsize=14)
    ax.set_ylabel(r"$Y\ \mathrm{[m]}$", fontsize=14)
    ax.set_title(r"Planform of $U$ (arrows: $U$, $V$)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.margins(x=0.15, y=0.05)
    ax.tick_params(labelsize=12)
    return ax


def instrument_position(d):
    """Y of the instrument face, from the first cell's position and its distance along the beam."""
    Y, dist = d["Y_m[m]"].to_numpy(), d["Distance_m"].to_numpy()
    direction = np.sign(Y[-1] - Y[0])  # +1: beams towards +Y (left bank), -1: towards -Y (right bank)
    return Y[0] - direction * dist[0]


def plot_beam_elevation(ADCPDataset, water_level, ax=None):
    """Cross-section view (Y horizontal, Z vertical): elevation of each ADCP beam below the
    water surface; cells coloured by U on the same scale as the planform map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    U_lim = np.nanmax(np.abs(ADCPDataset["U[m/s]"].to_numpy()))
    Y_all = ADCPDataset["Y_m[m]"].to_numpy()
    y_min, y_max = min(Y_all.min(), *[instrument_position(d) for _, d in ADCPDataset.groupby("Station")]), Y_all.max()

    ax.axhspan(ADCPDataset["Z_m[m]"].min() - 1, water_level, color=(0.30, 0.75, 0.93), alpha=0.12, linewidth=0)
    ax.axhline(water_level, color=(0.1, 0.4, 0.7), linewidth=1.5, linestyle="--")
    ax.text(y_max, water_level, rf"water surface (DSWL $= {water_level:.2f}$ m)", ha="right", va="bottom",
            fontsize=11, color=(0.1, 0.4, 0.7))

    for station, d in ADCPDataset.groupby("Station"):
        color = STATION_COLORS.get(station)
        Y, U, Z = d["Y_m[m]"].to_numpy(), d["U[m/s]"].to_numpy(), d["Z_m[m]"].iloc[0]
        ax.scatter(Y, np.full_like(Y, Z), c=U, cmap="RdBu_r", norm=TwoSlopeNorm(0, -U_lim, U_lim),
                   s=28, marker="s", edgecolors=color, linewidths=0.8, zorder=3)
        Y0 = instrument_position(d)
        ax.plot(Y0, Z, marker="D", markersize=8, color=color, markeredgecolor="k", zorder=4)
        # depth of the instrument below the water surface
        ax.annotate("", xy=(Y0, water_level), xytext=(Y0, Z),
                    arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.2))
        ax.text(Y0, Z + 0.015, f"  {station}: {water_level - Z:.2f} m  ", color=color,
                ha="left" if Y0 < Y.mean() else "right", va="center", fontsize=11, fontweight="bold")

    ax.set_xlim(y_min - 0.5, y_max + 0.5)
    ax.set_ylim(ADCPDataset["Z_m[m]"].min() - 0.08, water_level + 0.08)
    ax.set_xlabel(r"$Y\ \mathrm{[m]}$", fontsize=14)
    ax.set_ylabel(r"$Z\ \mathrm{[m]}$", fontsize=14)
    ax.set_title(r"ADCP beam elevation below the water surface (cells coloured by $U$; diamond = instrument)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    return ax


def plot_ADCP_U(ADCPDataset, water_level=None):
    """Lateral U profiles and planform map; with a water level, also the beam elevation view.

    water_level : water-surface elevation [m] (e.g. DSWL), in the same datum as Z_m[m].
    """
    if water_level is None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), num="ADCP U velocity",
                                       gridspec_kw={"width_ratios": [1.2, 1]})
        plot_U_profiles(ADCPDataset, ax=ax1)
        plot_U_planform(ADCPDataset, ax=ax2)
        fig.tight_layout()
        return fig

    fig = plt.figure(figsize=(18, 11), num="ADCP U velocity")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.5, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax3 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = fig.add_subplot(gs[:, 1])
    plot_U_profiles(ADCPDataset, ax=ax1, water_level=water_level)
    plot_beam_elevation(ADCPDataset, water_level, ax=ax3)
    plot_U_planform(ADCPDataset, ax=ax2)
    fig.tight_layout()
    return fig
