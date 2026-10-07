"""ANSYS Fluent results (Misery site) compared with the ADV June 2026 dataset.

Reads the Fluent report file (velocity, epsilon and discharge at the ADV points and cross-sections
against flow time), checks whether the run has reached a steady state, and compares the final
velocities with the time-averaged ADV velocities.

Run from the project folder (Misery/):
    poetry run python code/main_Fluent.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from main_ADV import W_d, Z_s, load_dataset

# Folder with one sub-folder per run (v_inlet_0.02, v_inlet_0.05, v_inlet_0.1)
fluentFolder = Path(__file__).resolve().parents[2] / "Fluent Simulation" / "Misery"
RUNS = {0.02: "v_inlet_0.02"}  # inlet velocity [m/s] -> folder (add 0.05 and 0.1 when available)
REPORT_FILE = "velocity-epsilon-discharge.out"

# ADV points shown in Figure 2 (one height per vertical)
FIG2_POINTS = ["x1y1z6", "x1y2z6", "x1y3z3", "x2y1z5", "x2y2z5", "x2y3z3", "x3y1z4", "x3y2z3", "x3y3z3"]
Q_ADV = (0.62, 0.74)  # discharge from the ADV verticals [m^3/s] (min, max)
Q_CRITICAL = 3.02     # discharge from the critical depth over the barrier [m^3/s]
AVG_WINDOW = 10.0     # length of the final window used as the "steady" value [s]

# Figure numbers in the documentation: inlet velocity -> (time series, ADV comparison)
FIGURE_NUMBERS = {0.1: (4, 5), 0.05: (6, 7), 0.02: (8, 9)}
figFolder = Path(__file__).resolve().parent.parent / "Doc_Figures"
SAVE_FIGURES = True


def read_report_out(outFile):
    """Fluent report file -> DataFrame indexed by flow time [s]; one column per report."""
    lines = Path(outFile).read_text().splitlines()
    header = next(l for l in lines if l.startswith("("))
    names = [n.strip('"') for n in header.strip("()").split('" "')]
    data = [list(map(float, l.split())) for l in lines if l and l[0].isdigit()]
    df = pd.DataFrame(data, columns=names)
    # a restarted run can repeat time steps: keep the last value of each
    return df.drop_duplicates("flow-time", keep="last").set_index("flow-time").sort_index()


def final_values(df, window=AVG_WINDOW):
    """Mean over the last `window` seconds, and its change from the window before (drift)."""
    t_end = df.index.max()
    last = df[df.index > t_end - window].mean()
    before = df[(df.index > t_end - 2 * window) & (df.index <= t_end - window)].mean()
    return last, last - before


def adv_mean_velocity(ParentDataset):
    """Time-averaged ADV streamwise velocity u [m/s] at every point (negative = downstream)."""
    return pd.Series({name: np.nanmean(ds["u"]) for name, ds in ParentDataset.items()})


def compare_points(final, U_adv, points):
    rows = []
    for p in points:
        U_cfd = final[f"velocity-{p}"]
        rows.append({"Point": p, "z_m": Z_s[int(p[5]) - 1], "U_ADV_m/s": U_adv[p], "U_Fluent_m/s": U_cfd,
                     "Fluent/ADV": U_cfd / U_adv[p]})
    return pd.DataFrame(rows)


def plot_time_series(df, v_inlet):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for p in FIG2_POINTS:
        axs[0].plot(df.index, df[f"velocity-{p}"], label=p)
    axs[0].plot(df.index, df["velocity-inlet"], "k--", label="inlet")
    axs[0].set(xlabel="flow time [s]", ylabel="$u$ [m/s]",
               title=f"Velocity at the ADV points (Figure 2 heights), $v_{{inlet}}$ = {v_inlet} m/s")
    axs[0].legend(ncol=2, fontsize=9)

    for col in ["inlet-discharge", "x1-discharge", "x2-discharge", "x3-discharge", "outlet-discharge"]:
        axs[1].plot(df.index, df[col].abs(), label=col)
    axs[1].axhspan(*Q_ADV, color="grey", alpha=0.3, label="ADV estimate")
    axs[1].axhline(Q_CRITICAL, color="k", ls=":", label="critical depth")
    axs[1].set(xlabel="flow time [s]", ylabel="$|Q|$ [m$^3$/s]", title="Discharge")
    axs[1].legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_profiles(final, U_adv, v_inlet):
    """Fluent and ADV velocity profiles at the center verticals (Y2)."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True)
    for k, ax in enumerate(axs, start=1):
        names = sorted(n for n in U_adv.index if n.startswith(f"x{k}y2z"))
        ax.plot(-U_adv[names], [Z_s[int(n[5]) - 1] for n in names], "o-", label="ADV")
        cfd = sorted(n for n in names if f"velocity-{n}" in final.index)
        ax.plot([-final[f"velocity-{n}"] for n in cfd], [Z_s[int(n[5]) - 1] for n in cfd], "s--",
                label=f"Fluent ($v_{{inlet}}$ = {v_inlet} m/s)")
        ax.axhline(W_d[k - 1][1], color="b", lw=0.8, label="water surface")
        ax.set(title=f"X{k}Y2 (center vertical)", xlabel="$-u$ [m/s] (downstream positive)")
    axs[0].set_ylabel("$z$ above the bed [m]")
    axs[0].legend(fontsize=9)
    fig.tight_layout()
    return fig


def main():
    ParentDataset = load_dataset()
    U_adv = adv_mean_velocity(ParentDataset)

    results = {}
    for v_inlet, folder in RUNS.items():
        df = read_report_out(fluentFolder / folder / REPORT_FILE)
        final, drift = final_values(df)
        print(f"\n=== Fluent run v_inlet = {v_inlet} m/s: {df.index.max():.1f} s simulated; "
              f"values averaged over the last {AVG_WINDOW:.0f} s ===")

        Q = pd.DataFrame({"Q_m3/s": final.filter(like="discharge"), "drift_m3/s": drift.filter(like="discharge")})
        print("--- Discharge (negative = towards -X) ---")
        print(Q.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"inlet velocity report: {final['velocity-inlet']:.3f} m/s "
              f"(set: {v_inlet} m/s); |Q_inlet| / inlet area 30.52 m^2 = {abs(final['inlet-discharge']) / 30.52:.3f} m/s")

        points = [c.removeprefix("velocity-") for c in df.columns
                  if c.startswith("velocity-x") and c.removeprefix("velocity-") in U_adv.index]
        Comparison = compare_points(final, U_adv, points)
        Comparison["drift_m/s"] = [drift[f"velocity-{p}"] for p in Comparison["Point"]]
        print("--- Fluent vs ADV velocity at the ADV points ---")
        print(Comparison.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        fig2 = Comparison[Comparison["Point"].isin(FIG2_POINTS)]
        print(f"Figure 2 points: Fluent/ADV = {fig2['Fluent/ADV'].median():.2f} (median), "
              f"{fig2['Fluent/ADV'].min():.2f}-{fig2['Fluent/ADV'].max():.2f} (range)")

        fig_t = plot_time_series(df, v_inlet)
        fig_p = plot_profiles(final, U_adv, v_inlet)
        if SAVE_FIGURES:
            for fig, n in zip((fig_t, fig_p), FIGURE_NUMBERS[v_inlet]):
                fig.savefig(figFolder / f"Figure_{n}.png", dpi=200, bbox_inches="tight")
        results[v_inlet] = {"reports": df, "final": final, "comparison": Comparison}

    plt.show()
    return results


if __name__ == "__main__":
    results = main()
