"""ADV June 2026 analysis (Misery site): mean velocity field, discharge, shear velocity.

Run from the project folder (Misery/):
    poetry run python code/main_ADV.py
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misery.discharge import depth_averaged_velocity, section_discharge
from misery.filters import mPST_ADVSpikeFilter
from misery.loglaw import KAPPA, fit_log_law
from misery.turbulence import reynolds_shear_stress
from misery.plotting_ADV import plot_3d_field, plot_planform

X = [-1.75, -6.75, -11.5]
Y1 = [2.14, 4.14, 6.14]
Y2 = [2.55, 4.55, 6.55]
Y3 = [2.92, 4.92, 6.92]
Z_b1 = [57.02, 57.02, 57.02]
Z_b2 = [57.01, 56.89, 57.09]
Z_b3 = [57.25, 57.23, 57.29]
Z_s = [0.02, 0.07, 0.17, 0.27, 0.37, 0.47, 0.57]
DSWL = 57.76 # downstream water level [m]
# Y of the water edge on the left bank at X1, X2, X3 [m]
Y_leftbank_edge = [-1.26, -0.85, +0.22]
# Y of the water edge on the right bank at X1, X2, X3 [m]
Y_rightbank_edge = [11.81, 11.23, 10.85]
# Water depth at each vertical [m] (rows: X1, X2, X3; columns: Y1, Y2, Y3 of that station)
W_d = [[0.83, 0.72, 0.64], [0.68, 0.6, 0.49], [0.48, 0.44, 0.52]]

# 1. Directory containing the files
dataFolder = Path(__file__).resolve().parent.parent / "dataset" / "ADV Data - June 2026"


def load_dataset():
    # 2. Find all .dat files in the directory (non-recursive, case-insensitive like Windows dir())
    datFiles = sorted(
        (p for p in dataFolder.iterdir() if p.is_file() and p.suffix.lower() == ".dat"),
        key=lambda p: p.name.lower(),
    )

    ParentDataset = {}
    # 3. Process each file
    for fullFilePath in datFiles:
        currentFileName = fullFilePath.name
        print(f"Importing: {currentFileName}...")

        # Whitespace-delimited numeric matrix, data starting at line 1 (no header)
        dataMatrix = (
            pd.read_csv(fullFilePath, sep=r"\s+", header=None)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        u = dataMatrix[:, 2]
        v = dataMatrix[:, 3]
        w = dataMatrix[:, 4]

        UVW_new = mPST_ADVSpikeFilter(u, v, w, 0)
        Sam_freq = 64

        # --- Parse filename coordinates ---
        # First 6 characters (e.g., 'x1y1z1' or 'x3y2z5')
        prefix = currentFileName[:6]
        xIdx = int(re.search(r"x(.*?)y", prefix).group(1))
        yIdx = int(re.search(r"y(.*?)z", prefix).group(1))
        zIdx = int(prefix.split("z", 1)[1])

        # Select X location
        xLoc = X[xIdx - 1]

        # Select Y location based on whether x is 3 or not
        if xIdx == 1:
            yLoc = Y1[yIdx - 1]
            zLoc = Z_b1[yIdx - 1] + Z_s[zIdx - 1]
        elif xIdx == 2:
            yLoc = Y2[yIdx - 1]
            zLoc = Z_b2[yIdx - 1] + Z_s[zIdx - 1]
        else:
            yLoc = Y3[yIdx - 1]
            zLoc = Z_b3[yIdx - 1] + Z_s[zIdx - 1]

        # --- Store in structure ---
        ParentDataset[prefix] = {
            "u": -UVW_new[:, 0],
            "v": -UVW_new[:, 1],
            "w": UVW_new[:, 2],
            "Sam_freq": Sam_freq,
            "location": np.array([xLoc, yLoc, zLoc]),
        }
    return ParentDataset


def compute_discharge(ParentDataset):
    """Discharge at each X station from the ADV verticals (velocity-area method).

    For each vertical, the time-averaged u at all its heights is depth-averaged
    (u = 0 at the bed, top value held constant to the surface); the unit discharges
    are then integrated across the section between the left and right water edges.
    """
    Y_stations = [Y1, Y2, Y3]
    rows = []
    for xIdx in range(1, 4):
        q_verticals = []
        for yIdx in range(1, 4):
            prefix = f"x{xIdx}y{yIdx}z"
            points = [(Z_s[int(name[5:]) - 1], np.nanmean(ds["u"]))
                      for name, ds in ParentDataset.items() if name.startswith(prefix)]
            z, u = zip(*points)
            ubar, q = depth_averaged_velocity(z, u, W_d[xIdx - 1][yIdx - 1])
            q_verticals.append(q)
        Q, segments = section_discharge(Y_stations[xIdx - 1], q_verticals,
                                        Y_leftbank_edge[xIdx - 1], Y_rightbank_edge[xIdx - 1])
        rows.append({
            "Station": f"X{xIdx}", "X_m": X[xIdx - 1],
            "Y_left_m": Y_leftbank_edge[xIdx - 1], "Y_right_m": Y_rightbank_edge[xIdx - 1],
            "q1": q_verticals[0], "q2": q_verticals[1], "q3": q_verticals[2],
            "Q_m3s": Q, "Edges_%": 100 * (segments[0] + segments[-1]) / Q,
        })

    DischargeTable = pd.DataFrame(rows)
    print("--- Discharge from ADV verticals (q in m^2/s, Q in m^3/s; negative = towards -X) ---")
    print(DischargeTable.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return DischargeTable


def compute_loglaw_fit(ParentDataset, yIdx=2, exclude=(), stations=(1, 2, 3)):
    """u_tau and z0 at each X station from the log law of the wall, fitted to the vertical yIdx
    (default 2, the center of the channel).

    z is the height above the bed (Z_s), not the global elevation. The ADV u is negative
    (flow towards -X), so -u is fitted. Points listed in exclude (e.g. "x1y2z1") are left out.
    """
    rows = []
    for xIdx in stations:
        prefix = f"x{xIdx}y{yIdx}z"
        points = sorted((Z_s[int(name[5:]) - 1], -np.nanmean(ds["u"]))
                        for name, ds in ParentDataset.items()
                        if name.startswith(prefix) and name not in exclude)
        z, U = map(np.array, zip(*points))
        u_tau, z0, r2 = fit_log_law(z, U)
        rows.append({
            "Vertical": f"X{xIdx}Y{yIdx}", "X_m": X[xIdx - 1], "Y_m": [Y1, Y2, Y3][xIdx - 1][yIdx - 1],
            "h_m": W_d[xIdx - 1][yIdx - 1], "N_points": len(z), "z_range_m": f"{z.min():.2f}-{z.max():.2f}",
            "u_tau_m/s": u_tau, "z0_m": z0, "ks_m (=30 z0)": 30 * z0, "R2": r2,
        })

    LogLawTable = pd.DataFrame(rows)
    print(f"--- Log-law fit U/u_tau = (1/kappa) ln(z/z0), kappa = {KAPPA}, z above the bed ---")
    print(LogLawTable.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return LogLawTable


def compute_reynolds_shear_velocity(ParentDataset, xIdx=1, yIdx=2, exclude=("x1y2z1",)):
    """u_tau from the Reynolds shear stress at each point of one vertical (default X1Y2).

    In uniform open-channel flow the shear stress falls linearly from the bed to the surface,
    -<u'w'> = u_tau^2 (1 - z/h), so each point gives u_tau = sqrt(-<u'w'> / (1 - z/h)).
    The streamwise velocity is -u (flow towards -X); the axes are rotated so that W = 0.
    """
    h = W_d[xIdx - 1][yIdx - 1]
    rows = []
    for name in sorted(ParentDataset):
        if not name.startswith(f"x{xIdx}y{yIdx}z") or name in exclude:
            continue
        ds = ParentDataset[name]
        z = Z_s[int(name[5:]) - 1]
        tau, tilt = reynolds_shear_stress(-ds["u"], ds["w"])
        rows.append({
            "Point": name, "z_m": z, "z/h": z / h, "U_m/s": -np.nanmean(ds["u"]), "tilt_deg": tilt,
            "-u'w'_m2/s2": tau, "u_tau_m/s": np.sqrt(max(tau, 0) / (1 - z / h)),
        })

    ReynoldsTable = pd.DataFrame(rows)
    print(f"--- u_tau from the Reynolds shear stress at X{xIdx}Y{yIdx} (h = {h} m) ---")
    print(ReynoldsTable.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return ReynoldsTable


def main():
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("--save-dir", type=Path, help="folder to save the figures as PNG")
    # parser.add_argument("--no-show", action="store_true", help="do not open figure windows")
    # args = parser.parse_args()

    ParentDataset = load_dataset()
    fig1 = plot_3d_field(ParentDataset, Y1, Y2, Y3)
    fig2, _ = plot_planform(ParentDataset, water_edges=(X, Y_leftbank_edge, Y_rightbank_edge))
    DischargeTable = compute_discharge(ParentDataset)
    LogLawTable = compute_loglaw_fit(ParentDataset)
    # Near-bed points with ~zero mean and low std (probe at the bed) or low beam correlation:
    # LogLawTable = compute_loglaw_fit(ParentDataset, exclude=("x1y2z1", "x2y2z1", "x2y2z2", "x3y2z1"))
    # Final estimate: X1Y2 only, without the near-bed point (bed in the sampling volume)
    LogLawX1Y2 = compute_loglaw_fit(ParentDataset, exclude=("x1y2z1",), stations=(1,))
    ReynoldsTable = compute_reynolds_shear_velocity(ParentDataset)

    # if args.save_dir:
    #     args.save_dir.mkdir(parents=True, exist_ok=True)
    #     fig1.savefig(args.save_dir / "Figure_1.png", dpi=200, bbox_inches="tight")
    #     fig2.savefig(args.save_dir / "Figure_2.png", dpi=200, bbox_inches="tight")
    # if not args.no_show:
    plt.show()
    return ParentDataset


if __name__ == "__main__":
    ParentDataset = main()
