# Misery — ADV & ADCP Field Data Analysis

Python tools to process and visualize velocity measurements from the **Misery** site of the
Sea Lamprey Barrier Project (field campaign of June 2026).

Two instruments are processed:

| Instrument | Data | What it gives |
|---|---|---|
| **ADV** — Nortek Vectrino | `.dat` time series, 64 Hz | 3D velocity (u, v, w) at points on a 3 × 3 grid of verticals, several heights each |
| **ADCP** — SonTek SL1500-3G (side-looking) | `.mat` export | Horizontal velocity (U, V) in 0.2 m cells across the channel, at one elevation |

## Repository layout

```
Misery/
├── code/
│   ├── main_ADV.py           # ADV: load, despike, time-average, plot
│   └── main_ADCP.py          # ADCP: load, average samples, orient, plot
├── src/misery/               # installable package (reusable functions)
│   ├── filters.py            # mPST_ADVSpikeFilter — ADV despiking
│   ├── discharge.py          # velocity–area discharge from verticals
│   ├── loglaw.py             # log-law fit: shear velocity u_tau and roughness z0
│   ├── turbulence.py         # Reynolds shear stress from ADV time series
│   ├── plotting_ADV.py       # 3D velocity field and planform plots (ADV)
│   ├── plotting_ADCP.py      # lateral U profiles, beam elevation and planform map (ADCP)
│   └── plotting_site.py      # site features shared by the plots (river water edges)
├── Doc_Figures/              # figures
├── pyproject.toml            # dependencies (Poetry)
├── LICENSE                   # CC BY-NC-SA 4.0
└── poetry.lock               # exact versions, for a reproducible environment
```

The field data (`dataset/`) are **not** included in this repository.

## Installation

Requires Python 3.12 and [Poetry](https://python-poetry.org/) (2.x).

```bash
git clone https://github.com/Roozbeh96/misery-adv-analysis.git
cd misery-adv-analysis
poetry install
```

`poetry install` creates a virtual environment in `.venv/` (see `poetry.toml`) and installs all
dependencies at the versions in `poetry.lock`, plus the `misery` package in editable mode, so
`from misery.filters import mPST_ADVSpikeFilter` works from any script.

## Data

Place the field data in a `dataset/` folder next to `code/`:

```
dataset/
├── ADV Data - June 2026/          # Vectrino files, e.g. x1y2z320260609120007.dat
└── ADCP Data - June 2026/
    ├── X1/X1_20260609_124914.mat
    ├── X2/X2_20260609_143925.mat
    └── X3/X3_20260609_134556.mat
```

ADV file names encode the sampling point: `x{i}y{j}z{k}…` = station *i* (X), vertical *j* (Y),
height *k* (Z). Only `.dat` files in the top level of the ADV folder are read.

## Usage

Run from the project folder:

```bash
poetry run python code/main_ADV.py
poetry run python code/main_ADCP.py
```

or activate the environment first (`source .venv/bin/activate`) and use `python …`.
In VS Code, select `.venv/bin/python` as the interpreter; the scripts can also be run in the
Interactive Window, where `ParentDataset` (ADV) and `ADCPDataset` (ADCP) stay in memory.

## Methods

### ADV (`main_ADV.py`)

1. **Read** each Vectrino `.dat` file; velocities are columns 3–5 (u, v, w).
2. **Despike** each component with `mPST_ADVSpikeFilter` (phase-space thresholding,
   Parsheh et al. 2010 / Wahl 2003; c1 = 1.8, c2 = 1.483).
3. **Orient**: u and v are sign-flipped (`-u`, `-v`) to match the site coordinate system.
4. **Locate** each point from its file name and the survey geometry (X stations, Y offsets,
   bed elevations, heights above the bed).
5. **Plot**:
   - 3D view of all sampling points, mean velocity vectors, bed and water surface;
   - planform of time-averaged velocity at selected heights, with a summary table and the
     surveyed water edges of both banks (`water_edges`).
6. **Discharge** (`compute_discharge`, velocity–area method):
   - each vertical is depth-averaged, `ū = (1/h) ∫ u dz`, with u = 0 at the bed and the top
     measured velocity held constant to the surface (trapezoidal rule); `q = ū·h`;
   - the unit discharges are integrated across the section, `Q = ∫ q dy`, with q = 0 at the
     water edges `Y_leftbank_edge` and `Y_rightbank_edge`, varying linearly in between.
7. **Shear velocity and bed roughness** at the center verticals (Y2):
   - `compute_loglaw_fit` fits the log law of the wall, `U/u_τ = (1/κ) ln(z/z0)` (κ = 0.39,
     z = height above the bed), by least squares in ln z, and reports u_τ, z0, `k_s ≈ 30 z0`
     and R². Unreliable points can be left out with `exclude` (e.g. `"x1y2z1"`).
   - `compute_reynolds_shear_velocity` estimates u_τ independently from the Reynolds shear
     stress, `−⟨u'w'⟩ = u_τ² (1 − z/h)`, after rotating the (u, w) axes at each point so that
     the mean vertical velocity is zero (removes the effect of probe tilt).

### ADCP (`main_ADCP.py`)

1. **Read** `Profile_X_Vel`, `Profile_Y_Vel` (mm/s → m/s), the per-beam standard deviations
   `Profile_0_VelStd`, `Profile_1_VelStd`, and the cell positions `Profile_Cell_Location`.
   Each file has 3 samples (rows, 90 s each) × up to 54 cells (columns).
2. **Combine the samples** (`SAMPLE_METHOD`): `"mean"` (default, average of all samples),
   `"min_row"` or `"min_per_cell"` (lowest-error sample).
3. **Error**: per cell, `Error = sqrt(σ0² + σ1²)`, propagated to the mean as
   `sqrt(mean(Error²) / N)`. Component errors follow from the beam geometry:

   ```
   σ_U = Error / (2 sin θ),   σ_V = Error / (2 cos θ),   θ = 25.37°
   ```

   The beam angle θ was determined from the data: `Profile_X_Vel = (b0 − b1) / (2 sin θ)` holds
   to within 0.02 mm/s in all files.
4. **Orient** (`Orientation`): instruments on the left bank (X1, X2, facing downstream) keep their
   axes; the instrument on the right bank (X3) is turned 180°, so U → −U, V → −V and its cells run
   toward decreasing Y.
5. **Locate** each cell: `Y = Y_instrument + Orientation · distance`, with station X and the
   beam elevation Z, both in the global (survey) coordinate system. The downstream water level
   `DSWL = 57.76 m` gives the depth of each instrument below the surface (0.10, 0.13, 0.24 m).
6. **Plot** (`plotting_ADCP.plot_ADCP_U(ADCPDataset, water_level=DSWL)`):
   - lateral profiles of U with ±σ_U bands (legend: position and depth below the surface);
   - beam elevation view (Y–Z): water surface, each instrument and its beam, cells coloured by U;
   - planform map coloured by U with (U, V) arrows, the instrument positions and the
     surveyed water edges of both banks (`water_edges`).

## Notes and known limitations

- ADCP cells farther than about 7–8 m from the instrument are noisy (weak echo, near the far bank).
- The ADCP error bands include instrument noise only; the scatter between the three samples is
  larger, so the true uncertainty of U is higher.
- The ADCP beams are at different depths below the surface (0.10–0.24 m), so the U profiles of
  the three stations are not taken at the same relative depth.
- Discharge: with only three verticals per section, 33–44% of Q comes from the segments
  between the outer verticals and the water edges, where q is assumed to fall linearly to zero.
- Log law: several near-bed ADV points are unreliable (bed inside the sampling volume, or beam
  correlation below 70%) and strongly affect the fit. X1Y2 is the only vertical with enough
  reliable points (z = 0.07–0.47 m); there the log law gives u_τ ≈ 0.020 m/s and z0 ≈ 0.0045 m,
  and the Reynolds stress (z ≤ 0.27 m) gives u_τ ≈ 0.014 m/s. Above z ≈ 0.3 m the shear stress
  increases with height (flow downstream of the barrier is not uniform), so those points are
  not used.

## Work in progress

- Combined ADV + ADCP analysis in global coordinates.

## License

This work is licensed under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0); see [LICENSE](LICENSE).
You may share and adapt it for non-commercial purposes, with attribution, under the same license.
