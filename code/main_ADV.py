"""Python port of main.m — ADV June 2026 mean velocity field (Misery site).

Run from the project folder (Misery/):
    poetry run python code/main.py                 # show figures
    poetry run python code/main.py --save-dir out  # also save PNGs
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misery.filters import mPST_ADVSpikeFilter
from misery.plotting_ADV import plot_3d_field, plot_planform

X = [-1.75, -6.75, -11.5]
Y1 = [2.14, 4.14, 6.14]
Y2 = [2.55, 4.55, 6.55]
Y3 = [2.92, 4.92, 6.92]
Z_b1 = [57.02, 57.02, 57.02]
Z_b2 = [57.01, 56.89, 57.09]
Z_b3 = [57.25, 57.23, 57.29]
Z_s = [0.02, 0.07, 0.17, 0.27, 0.37, 0.47, 0.57]
DSWL = 57.76
Y_leftbank_edge = [-1.26, -0.85, +0.22]

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


def main():
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("--save-dir", type=Path, help="folder to save the figures as PNG")
    # parser.add_argument("--no-show", action="store_true", help="do not open figure windows")
    # args = parser.parse_args()

    ParentDataset = load_dataset()
    fig1 = plot_3d_field(ParentDataset, Y1, Y2, Y3)
    fig2, _ = plot_planform(ParentDataset)

    # if args.save_dir:
    #     args.save_dir.mkdir(parents=True, exist_ok=True)
    #     fig1.savefig(args.save_dir / "Figure_1.png", dpi=200, bbox_inches="tight")
    #     fig2.savefig(args.save_dir / "Figure_2.png", dpi=200, bbox_inches="tight")
    # if not args.no_show:
    plt.show()
    return ParentDataset


if __name__ == "__main__":
    ParentDataset = main()
