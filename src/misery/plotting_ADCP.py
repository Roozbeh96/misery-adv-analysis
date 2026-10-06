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


def plot_U_profiles(ADCPDataset, ax=None, theta_deg=BEAM_ANGLE_DEG):
    """U across Y for each station, with a shaded +/- U error band (instrument error)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    for station, d in ADCPDataset.groupby("Station"):
        Y = d["Y_m[m]"].to_numpy()
        U = d["U[m/s]"].to_numpy()
        U_err = d["Error[m/s]"].to_numpy() / (2 * np.sin(np.radians(theta_deg)))
        color = STATION_COLORS.get(station)
        label = rf"{station}  ($X = {d['X_m[m]'].iloc[0]:.2f}$ m, $Z = {d['Z_m[m]'].iloc[0]:.2f}$ m)"

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


def plot_ADCP_U(ADCPDataset):
    """Figure with the lateral U profiles (left) and the planform map (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), num="ADCP U velocity",
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    plot_U_profiles(ADCPDataset, ax=ax1)
    plot_U_planform(ADCPDataset, ax=ax2)
    fig.tight_layout()
    return fig
