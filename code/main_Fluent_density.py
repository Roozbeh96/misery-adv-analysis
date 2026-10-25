"""Side-density (free-surface) comparison of the Fluent runs with different critical depths (Misery site).

The input images are Fluent screenshots of the mixture density on the side plane, exported with different
camera views. They are put on a common scale using the geometry:
    - horizontal origin: the downstream face of the barrier crest; vertical origin: the crest (58.79 m),
    - relative zoom between images: the distance from the crest to the small block on the downstream bed,
    - metres per pixel: least-squares fit to the prescribed water depths over the crest (y_c).
The free surface is taken where the density colour crosses the middle of the colour map (about 500 kg/m^3).

Run from the project folder (Misery/):
    poetry run python code/main_Fluent_density.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.ndimage import median_filter

fluentFolder = Path(__file__).resolve().parents[2] / "Fluent Simulation" / "Misery"
figFolder = Path(__file__).resolve().parent.parent / "Doc_Figures"
SAVE_FIGURES = True
FIGURE_NUMBERS = (14, 15)  # (aligned density contours, free-surface profiles)

Z_CREST = 58.79  # barrier crest [m]
# image, critical depth y_c [m] (= inlet level - crest), discharge through X1-X3 [m^3/s], colour-bar mask (x < px)
RUNS = [
    ("v_inlet_0.1/density_profile_0.19.png", 0.19, 3.18, 0),
    ("y_c_0.12_58.91/density_profile_0.12.png", 0.12, 1.88, 0),
    ("y_c_0.07_58.86/density_profile_0.07.png", 0.07, 1.13, 140),
]
X_LIM = (-4.0, 6.5)    # common window, distance from the barrier crest [m]
Z_LIM = (56.6, 59.3)   # elevation [m]
RHO = (1.22, 998.0)    # density range of the Fluent colour map [kg/m^3]


def detect(im, x_mask=0, y_max=600):
    """Crest, block and free surface in one screenshot (pixel coordinates, y downwards)."""
    sat = im.max(axis=2) - im.min(axis=2) > 0.3   # coloured = inside the fluid domain
    sat[y_max:] = False                            # horizontal colour bar
    sat[:, :x_mask] = False                        # vertical colour bar
    water = sat & (im[..., 0] > im[..., 2])        # red > blue: density above the middle of the map
    W = im.shape[1]
    bottom = np.array([np.where(sat[:, x])[0].max() if sat[:, x].any() else -1 for x in range(W)])

    mid = np.arange(W // 4, 3 * W // 4)
    y_crest = bottom[mid].min()
    x_crest = mid[bottom[mid] <= y_crest + 2].max()        # downstream face of the crest
    ds = np.arange(x_crest + 60, W - 5)
    x_block = ds[np.argmin(bottom[ds])]                    # small block on the downstream bed

    surface = np.full(W, np.nan)
    for x in range(x_mask, W):
        rows = np.where(sat[:, x])[0]
        if len(rows):
            col = water[:, x]
            run = np.where(np.convolve(col[rows[0]:y_max], np.ones(5), "valid") == 5)[0]
            if len(run):
                surface[x] = rows[0] + run[0]
    return {"x_crest": x_crest, "y_crest": y_crest, "x_block": x_block, "surface": surface, "x_mask": x_mask}


def calibrate(dets):
    """Pixels per metre of each image: relative zoom from the crest-block distance, absolute scale from y_c."""
    L = np.array([d["x_block"] - d["x_crest"] for d in dets], float)
    zoom = L / L[0]
    x_up = [slice(d["x_mask"] + 20, d["x_crest"] - 150) for d in dets]
    head_px = np.array([d["y_crest"] - np.nanmedian(d["surface"][s]) for d, s in zip(dets, x_up)]) / zoom
    y_c = np.array([r[1] for r in RUNS])
    s0 = np.sum(head_px * y_c) / np.sum(y_c ** 2)          # least squares: head_px = s0 * y_c
    return s0 * zoom


def to_metres(d, s, shape):
    """imshow extent [x0, x1, z0, z1] and the free-surface profile in metres."""
    H, W = shape[:2]
    extent = [(0 - d["x_crest"]) / s, (W - d["x_crest"]) / s, Z_CREST - (H - d["y_crest"]) / s,
              Z_CREST + d["y_crest"] / s]
    x = (np.arange(W) - d["x_crest"]) / s
    z = Z_CREST - (median_filter(d["surface"], 5, mode="nearest") - d["y_crest"]) / s
    return extent, x, z


def plot_contours(images, dets, scales):
    fig, axs = plt.subplots(len(RUNS), 1, figsize=(11, 8.5), sharex=True, sharey=True)
    for ax, im, d, s, (_, y_c, Q, _) in zip(axs, images, dets, scales, RUNS):
        extent, _, _ = to_metres(d, s, im.shape)
        img = im.copy()
        img[:, :d["x_mask"]] = 1.0                       # hide the vertical colour bar
        ax.imshow(img, extent=extent, aspect="equal", interpolation="bilinear")
        ax.axhline(Z_CREST + y_c, color="k", ls="--", lw=0.8)
        ax.set_title(f"$y_c$ = {y_c:.2f} m (inlet level {Z_CREST + y_c:.2f} m), $Q$ = {Q:.2f} m$^3$/s", fontsize=11)
        ax.set(xlim=X_LIM, ylim=Z_LIM, ylabel="$z$ [m]")
    axs[-1].set_xlabel("distance from the barrier crest [m] (downstream positive)")
    sm = plt.cm.ScalarMappable(norm=Normalize(*RHO), cmap="jet")
    fig.colorbar(sm, ax=axs, shrink=0.6, label="density (mixture) [kg/m$^3$]")
    return fig


def plot_surfaces(images, dets, scales):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["tab:red", "tab:orange", "tab:blue"]
    for im, d, s, (_, y_c, Q, _), c in zip(images, dets, scales, RUNS, colors):
        _, x, z = to_metres(d, s, im.shape)
        ok = (x >= X_LIM[0]) & (x <= X_LIM[1]) & (np.arange(len(x)) >= d["x_mask"] + 5)
        ax.plot(x[ok], z[ok], color=c, lw=1.6, label=f"$y_c$ = {y_c:.2f} m, $Q$ = {Q:.2f} m$^3$/s")
    ax.axhline(Z_CREST, color="grey", lw=0.8)
    ax.text(X_LIM[0] + 0.1, Z_CREST - 0.07, "crest 58.79 m", color="grey", fontsize=9)
    ax.axvline(0, color="grey", lw=0.8, ls=":")
    ax.set(xlim=X_LIM, xlabel="distance from the barrier crest [m] (downstream positive)",
           ylabel="free-surface elevation $z$ [m]", title="Free surface (density = 500 kg/m$^3$) on the side plane")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def main():
    images = [plt.imread(fluentFolder / f)[..., :3] for f, *_ in RUNS]
    dets = [detect(im, x_mask) for im, (*_, x_mask) in zip(images, RUNS)]
    scales = calibrate(dets)

    print("--- Image calibration ---")
    for d, s, (f, y_c, _, _) in zip(dets, scales, RUNS):
        i_up = slice(d["x_mask"] + 20, d["x_crest"] - 150)
        head = (d["y_crest"] - np.nanmedian(d["surface"][i_up])) / s
        _, x, z = to_metres(d, s, (0, len(d["surface"])))
        far = (x > X_LIM[1] - 1.0) & (x <= X_LIM[1])
        print(f"{f}: {s:.1f} px/m; depth over crest upstream {head:.3f} m (set {y_c:.2f} m); "
              f"free surface {X_LIM[1] - 0.5:.1f} m downstream: {np.nanmedian(z[far]):.2f} m")

    fig1 = plot_contours(images, dets, scales)
    fig2 = plot_surfaces(images, dets, scales)
    if SAVE_FIGURES:
        for fig, n in zip((fig1, fig2), FIGURE_NUMBERS):
            fig.savefig(figFolder / f"Figure_{n}.png", dpi=200, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
