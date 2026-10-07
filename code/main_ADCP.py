"""ADCP June 2026 analysis (Misery site) — SonTek SL1500-3G side-looking profiler.

Each station file (X1, X2, X3) holds:
    rows    = consecutive flow samples (90 s each)
    columns = measurement cells along the beam, at Profile_Cell_Location [mm]
              from the instrument face (horizontal, across the channel)

Run from the project folder (Misery/):
    poetry run python code/main_ADCP.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from misery.plotting_ADCP import plot_ADCP_U

# 1. Directory containing the ADCP files (one sub-folder per station: X1, X2, X3)
dataFolder = Path(__file__).resolve().parent.parent / "dataset" / "ADCP Data - June 2026"

X = [-1.75, -6.75, -11.5]
Z = [57.66, 57.63, 57.52]
DSWL = 57.76  # downstream water level [m]
# Y of the water edge on the left bank at X1, X2, X3 [m]
Y_leftbank_edge = [-1.26, -0.85, +0.22]
# Y of the water edge on the right bank at X1, X2, X3 [m]
Y_rightbank_edge = [11.81, 11.23, 10.85]

# Instrument orientation (facing downstream): +1 = left bank, -1 = right bank.
# On the right bank the instrument is turned 180 deg about the vertical axis, so its X and Y
# axes are reversed: U -> -U, V -> -V, and its cells run towards decreasing Y.
Orientation = [1, 1, -1]

# Global Y of each cell: Y = Y_instrument + Orientation * cell distance from the instrument
Y_instrument = [-1.13, -0.34, 9.91]  # across-channel position of the instrument [m]

# How to reduce the samples (rows) to one value per cell:
#   "min_row"      : the single sample (row) with the lowest mean error
#   "min_per_cell" : for each cell, the sample with the lowest error
#   "mean"         : average of all samples
SAMPLE_METHOD = "mean"


def read_adcp_file(matFile):
    """Read one SL1500 .mat file; velocities and errors converted to m/s, distances to m."""
    m = sio.loadmat(matFile, squeeze_me=True)
    return {
        "Vx": np.atleast_2d(m["Profile_X_Vel"]) / 1000,
        "Vy": np.atleast_2d(m["Profile_Y_Vel"]) / 1000,
        # Combined std of the two beams; the X and Y velocity errors are both proportional to it
        "error": np.sqrt(np.atleast_2d(m["Profile_0_VelStd"]) ** 2
                         + np.atleast_2d(m["Profile_1_VelStd"]) ** 2) / 1000,
        "cell_distance": np.atleast_2d(m["Profile_Cell_Location"])[0] / 1000,
    }


def select_samples(Vx, Vy, error, method=SAMPLE_METHOD):
    """Reduce (samples x cells) arrays to one value per cell. Returns U, V, error, sample used."""
    nSamples, nCells = Vx.shape
    cells = np.arange(nCells)
    if method == "min_row":
        best = np.nanargmin(np.nanmean(error, axis=1))
        rows = np.full(nCells, best)
    elif method == "min_per_cell":
        rows = np.nanargmin(error, axis=0)
    elif method == "mean":
        # error of the mean of nSamples independent samples
        return (np.nanmean(Vx, axis=0), np.nanmean(Vy, axis=0),
                np.sqrt(np.nanmean(error**2, axis=0) / nSamples), np.full(nCells, -1))
    else:
        raise ValueError(f"Unknown method: {method}")
    return Vx[rows, cells], Vy[rows, cells], error[rows, cells], rows + 1  # sample number, 1-based


def load_dataset(method=SAMPLE_METHOD):
    stations = []
    for k, station in enumerate(["X1", "X2", "X3"]):
        matFile = sorted((dataFolder / station).glob("*.mat"))[0]
        print(f"Importing: {matFile.name}...")
        d = read_adcp_file(matFile)
        U, V, err, sample = select_samples(d["Vx"], d["Vy"], d["error"], method)

        nCells = len(U)
        stations.append(
            pd.DataFrame(
                {
                    "Station": station,
                    "Cell": np.arange(1, nCells + 1),
                    "Distance_m": d["cell_distance"],  # from the instrument face
                    "X_m[m]": X[k],
                    "Y_m[m]": Y_instrument[k] + Orientation[k] * d["cell_distance"],
                    "Z_m[m]": Z[k],
                    "U[m/s]": Orientation[k] * U,  # Profile_X_Vel [m/s], in global X
                    "V[m/s]": Orientation[k] * V,  # Profile_Y_Vel [m/s], in global Y
                    "Error[m/s]": err,  # combined beam std [m/s]
                    "Sample": sample,  # sample (row) used; -1 = mean of all samples
                }
            )
        )
    return pd.concat(stations, ignore_index=True)


def main():
    ADCPDataset = load_dataset()
    print(f"--- ADCP velocities per cell (method: {SAMPLE_METHOD}) ---")
    print(ADCPDataset.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig = plot_ADCP_U(ADCPDataset, water_level=DSWL, water_edges=(X, Y_leftbank_edge, Y_rightbank_edge))
    plt.show()
    return ADCPDataset


if __name__ == "__main__":
    ADCPDataset = main()
