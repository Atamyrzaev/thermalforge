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
  of the die. On an 8 W, 5 × 5 mm NPU, assuming uniform power under-reports the
  junction by **13.2 K**. Four built-in patterns cover the common shapes, and
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
| **Not validated** | No result has been compared with a measured package. Until that happens, every reported temperature is labelled *verified, not validated* (`src/validation_status.py`). |
| **Ready for measurements** | `tools/correlate.py` takes a case file — a package plus the temperatures somebody measured on it — and reports agreement against two bands: the instrument's uncertainty and the model's error budget. A synthetic example ships so the process can be rehearsed today; everything derived from it is stamped SYNTHETIC. |
| **Error budget** | Two measured sensitivities, added in quadrature and printed beside every peak: mesh refinement (`tools/mesh_convergence.py`: ±0.43 K passive, ±1.76 K cold plate) and conductivity perturbations (`tools/material_sensitivity.py`: silicon ±10 %, bump composite ±30 %). On the demo scenarios that is **±2.1 to ±2.6 K, about 3.5 % of the rise** — dominated by the silicon conductivity, not by the mesh. Results are versioned in `validation/`. |

Known limitations are printed on the reports themselves, not hidden here: power
enters at the active plane but is spread over one mesh cell (20 µm) rather than a
sub-micron device layer, silicon is homogeneous (no BEOL, device layer or TSVs),
cooling is one effective coefficient on the die backside, the stack is three
editable layers (interposer / bumps / die) rather than an arbitrary one, and the
keep-out is enforced in whole mesh cells, so a 100 µm rule becomes 500 µm on a
0.5 mm mesh — stricter than asked, never looser.

## Layout

| Path | What it is |
|---|---|
| `studio/` | The app: `engine.py` (physics + exact optimizer), `server.py` (stdlib HTTP API), `static/` (the UI) |
| `src/` | The core: `thermal_solver.py`, `grid_builder.py`, `composite_materials.py`, `experiment.py`, `exhaustive.py`, `optimizer.py`, `surrogate.py` |
| `main.py` | Command-line pipeline (Stage 1 single die, Stage 2 placement) |
| `packaging/` | The standalone build: `build.py` and the PyInstaller spec |
| `tools/` | `mesh_convergence.py` and `material_sensitivity.py` (the error budget), `correlate.py` (measured against modelled), `smoke.py` (does a build start and answer?) |
| `validation/` | Verification artifacts, versioned as evidence: the studies, plus `cases/` and `correlation/` |
| `examples/` | Power maps to import: one matrix form, one region table |
| `tests/` | 298 tests: `test_core.py`, `test_studio.py`, `test_benchmarks.py` |

## Development

```bash
pip install -r requirements.txt
python -m pytest -q
```

The three protected modules — `src/thermal_solver.py`, `src/composite_materials.py`
and `src/exhaustive.py` — carry the numerical core; change them only with a test
that fails first.
