# RISC3D-AI

Thermal co-optimization for 3D-IC chiplet packages: a verified steady-state heat
solver, and a placement search that returns the **provably coolest** layout on a
stated grid rather than a good guess.

```bash
python run_studio.py
```

That opens **RISC3D Studio** in your browser at `http://127.0.0.1:8765`. On
Windows, double-clicking `run_studio.bat` does the same. Nothing is downloaded and
no network is used: the whole app is stdlib Python plus NumPy/SciPy, with vanilla
JavaScript in the browser.

### On a machine with no Python

```bash
python packaging/build.py --check
```

builds `dist/RISC3D Studio/` — about 125 MB holding Python, NumPy, SciPy, the
solver, the UI and the measured validation artifacts. Copy that folder anywhere and
double-click **RISC3D Studio.exe**. Add `--zip` for one 52 MB file to hand over;
the folder carries a `README_FIRST.txt` explaining the first-launch warning. `--check` then starts the built app and makes it
solve a scenario over HTTP, because a build that imports cleanly can still be missing
a file it only reads at runtime.

First launch takes about 8 seconds while Windows reads 125 MB of fresh files; after
that it is about 1 second. The console window is the server — closing it stops the
app — and starting a second copy just opens the one already running. Requires
PyInstaller (`pip install pyinstaller`).

The executable is **unsigned**, so the first launch on a fresh machine shows
"Windows protected your PC": *More info → Run anyway*. Signing it is a purchase, not
a code change.

## What it does

- **Solves the package in 3-D.** Finite-volume steady-state conduction through the
  silicon interposer, the microbump/underfill layer and the die, cooled through
  the die backside. Every solve checks its own residual and energy balance and
  raises rather than reporting a number it cannot defend.
- **Models what is inside a chiplet.** A part is not a uniform heater: an NPU's MAC
  array, a GPU's engines and a CPU's core complex carry most of the power over part
  of the die. On an 8 W, 5 × 5 mm NPU alone on a 20 mm package at h = 1,000, assuming
  uniform power under-reports the junction by **12.5 K** (85.6 °C against 98.1 °C).
  Each block takes the cells it geometrically overlaps rather than the cells it
  rounds to, so the modelled hotspot is the same size at every mesh. Four built-in
  patterns cover the common shapes, and
  **Import map…** reads a design team's own numbers from a CSV — either a matrix of
  tile values or a table of regions (see `examples/`). Values are relative, so watts,
  milliwatts, percentages or activity counts all work.
- **Returns layouts that can be built.** A minimum chiplet-to-chiplet clearance
  (default 100 µm, for underfill flow and bump dispense tolerance) constrains the
  search, so the proven optimum is the coolest *manufacturable* layout — including
  refusing corner-to-corner contact.
- **Re-solves as you drag.** The expensive matrix factorization is computed once
  per package and reused, so moving a chiplet costs milliseconds.
- **Proves the optimum.** Heat adds up linearly, so a layout's temperature is the
  sum of what each chiplet causes alone. Branch-and-bound with a lower bound that
  is *proven* (a chiplet can only add heat) covers every layout on the placement
  grid and skips only branches that cannot contain a better one. The winner is
  re-solved from scratch and refused if the two disagree by more than 1e-6 K.
  The search runs its root branches on every core and splits its batched solves
  across threads — with ties broken by the layout rather than by which thread
  arrived first, so the answer never depends on how the work was divided.
- **Answers over time, not just at rest.** A burst — idle, full power, idle —
  time-stepped through the same solver, with the steady-state asymptote, the
  limit line and the moment it was crossed. A part that settles at 86 °C can
  still cross 80 °C **592 ms** into a one-second burst, and a lid's thermal mass
  can carry another through it entirely.
- **Shows its work.** Before/after maps on one scale, every chiplet's move, the
  proof figures, the solver checks, and a self-contained HTML report.

## Status: verified, not validated

| | |
|---|---|
| **Verified** | Residual and energy balance on every solve. The optimizer agrees with an independent brute force. The solver reproduces closed-form solutions for a uniformly heated slab (second order, with the predicted coefficient) and a rectangular flux channel (`tests/test_benchmarks.py`). |
| **Partly validated** | The *spreading* has been compared with a measured thermal test chip (Terraneo et al., SAMOS 2019, CC-BY): 16 on-die diodes, **28.95 K of measured spatial spread reproduced to 0.63 K RMS**, with the two unpublished inputs fitted on a *different* power pattern and held fixed. The *level* has not: the whole map sits 2.80 K low together, which is a series resistance outside this model, and the convective coefficient was fitted rather than measured. Every reported temperature is labelled *verified; spreading validated, absolute level not validated* (`src/validation_status.py`). |
| **Ready for measurements** | `tools/correlate.py` takes a case file — a package plus the temperatures somebody measured on it — and reports agreement against two bands: the instrument's uncertainty and the model's error budget. It splits the disagreement in two, because the halves mean different things: a **common offset** on every sensor is an unmodelled series resistance (a contact, a grease line, a heat sink nobody disclosed) and one number removes it; what is left is **shape**, which is the conduction this solver actually computes. It refuses to call a case validating when that case cannot discriminate — one measurement, or several agreeing within their own uncertainty, would be matched by a wrong model too. A synthetic example ships so the process can be rehearsed; everything derived from it is stamped SYNTHETIC. |
| **Measured data found so far** | Two real, independently measured candidates are documented in [`validation/MEASURED_DATA.md`](validation/MEASURED_DATA.md), with what each can and cannot prove. The first is dominated by interface resistances this model does not compute (≈9 % of it is the silicon conduction we solve), so it can calibrate but not confirm. The second — an open CC-BY thermal-test-chip dataset, 13 power patterns × 3 cooling conditions with 16-point temperature maps — can discriminate, and is the one being worked. Neither has yet changed the status below. |
| **Error budget** | Two measured sensitivities, added in quadrature and printed beside every peak. **Mesh**: reference packages refined to 0.25 mm and compared directly, so the discretization error is measured rather than extrapolated. Six reference cases span what the app permits — spread-out, concentrated and internally patterned chiplets, each passively cooled and under a cold plate — because what drives the error is how concentrated the heat is. A design is banded by the worst case in the family it belongs to: 1.34 % of the rise for uniform chiplets, 2.14 % with internal patterns. **Materials**: silicon ±10 %, bump composite ±30 % (`tools/material_sensitivity.py`). **Boundary**: the convective coefficient ±9 %, which is how much a real rig's own `h` varies across 13 power patterns (`tools/boundary_sensitivity.py`) — and it is the **largest** of the three at about 5 % of the rise, so the answer is limited by how well the cooling is known, not by the solver. On the demo scenarios: **±4.08, ±3.19 and ±2.34 K**. Results are versioned in `validation/`. |
| **And the band is audited** | `tools/band_check.py` solves fourteen designs at both mesh levels and compares what it measures against the band the app quoted them. That is how the reference set was found wanting: four designs exceeded their band, the worst by 2.85×, every one of them a uniform design with a small dense part the reference cases did not represent. With those cases added, all fourteen come in under — worst ratio 0.81, on the design that used to break it. A band that flatters is worse than no band, because somebody spends it. |
| **Finer when you want it** | The mesh is a choice, not a ceiling. Each level takes the finest cell that fits its unknown budget: **Standard** 36,000 (0.5 mm on a 20 mm package, 0.25 mm on a 10 mm one, answers in under a second) or **Fine** 120,000 (0.25 mm on a 20 mm package, 4–25 s to build depending on the cooling stack). The factorization is exact at either size — residuals stay near 1e-11 — so a finer mesh costs time and memory, not correctness. |
| **Turn parts, if it helps** | The exact search can take 90° rotations as candidate states (`"rotate": true` on the package): a part's sides swap and its internal pattern turns with them, checked against a brute force that rotates independently. Off by default on measured grounds — on all three demo scenarios it moves the proven optimum **0.000 K** while doubling the candidate count, and it only earns its cost where packing is tight (0.156 K on three long thin parts). |
| **Or stop estimating** | **Measure this design's mesh error** solves the design on screen at both mesh levels and subtracts. That is this design's discretization error, measured, rather than a figure scaled from a reference package. On the three demo scenarios it comes to 0.062, 0.046 and 0.013 K against estimates of 1.30, 1.09 and 0.75 K. The estimate has to hold for every design, so it is conservative for most; the measurement is what to quote when a design sits near its limit. |

Known limitations are printed on the reports themselves, not hidden here: power
enters at the active plane but is spread over one mesh cell (20 µm) rather than a
sub-micron device layer, silicon is homogeneous (no BEOL, device layer or TSVs),
cooling is one effective coefficient on the die backside, the stack is three
editable layers (interposer / bumps / die) rather than an arbitrary one, and the
keep-out is enforced in whole mesh cells, so a 100 µm rule becomes 500 µm on a
0.5 mm mesh — stricter than asked, never looser, and halved by a finer one.

One limitation is enforced rather than printed. A hotspot smaller than a mesh
cell is averaged over that cell, so the reported peak stops depending on the
design: measured, blocks of 0.50, 0.25 and 0.10 mm inside a 5 mm part all
report the same temperature, and refining from 0.5 mm to 0.25 mm raises the
0.50 mm case by **29.2 K** against a quoted band of 1.98 K. The app detects
this, fails its **Hotspots resolved** check, and declines to quote a mesh band
it knows does not apply — the peak is labelled a lower bound. Resolving a
0.1 mm hotspot would need 0.05 mm cells, which no mesh this app builds
provides; that is what local refinement is for, and it is not built yet.

## Layout

| Path | What it is |
|---|---|
| `studio/` | The app: `engine.py` (physics + exact optimizer), `server.py` (stdlib HTTP API), `static/` (the UI) |
| `src/` | The core: `thermal_solver.py`, `grid_builder.py`, `composite_materials.py`, `experiment.py`, `exhaustive.py`, `optimizer.py`, `surrogate.py` |
| `main.py` | Command-line pipeline (Stage 1 single die, Stage 2 placement) |
| `packaging/` | The standalone build: `build.py` and the PyInstaller spec |
| `tools/` | `mesh_convergence.py` and `material_sensitivity.py` (the error budget), `band_check.py` (is the band actually a bound?), `correlate.py` (measured against modelled), `smoke.py` (does a build start and answer?) |
| `validation/` | Verification artifacts, versioned as evidence: the studies, plus `cases/` and `correlation/` |
| `examples/` | Power maps to import: one matrix form, one region table |
| `tests/` | 373 tests: `test_core.py`, `test_studio.py`, `test_benchmarks.py` |

## Development

```bash
pip install -r requirements.txt
python -m pytest -q
```

The three protected modules — `src/thermal_solver.py`, `src/composite_materials.py`
and `src/exhaustive.py` — carry the numerical core; change them only with a test
that fails first.
