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

## What it does

- **Solves the package in 3-D.** Finite-volume steady-state conduction through the
  silicon interposer, the microbump/underfill layer and the die, cooled through
  the die backside. Every solve checks its own residual and energy balance and
  raises rather than reporting a number it cannot defend.
- **Models what is inside a chiplet.** A part is not a uniform heater: an NPU's MAC
  array, a GPU's engines and a CPU's core complex carry most of the power over part
  of the die. On an 8 W, 5 × 5 mm NPU, assuming uniform power under-reports the
  junction by **13.2 K**.
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
- **Shows its work.** Before/after maps on one scale, every chiplet's move, the
  proof figures, the solver checks, and a self-contained HTML report.

## Status: verified, not validated

| | |
|---|---|
| **Verified** | Residual and energy balance on every solve. The optimizer agrees with an independent brute force. The solver reproduces closed-form solutions for a uniformly heated slab (second order, with the predicted coefficient) and a rectangular flux channel (`tests/test_benchmarks.py`). |
| **Not validated** | No result has been compared with a measured package. Until that happens, every reported temperature is labelled *verified, not validated* (`src/validation_status.py`). |
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
| `tools/` | `mesh_convergence.py` and `material_sensitivity.py` (the error budget), `smoke.py` (does a build start and answer?) |
| `validation/` | Measured verification artifacts, versioned as evidence |
| `tests/` | 270 tests: `test_core.py`, `test_studio.py`, `test_benchmarks.py` |

## Development

```bash
pip install -r requirements.txt
python -m pytest -q
```

The three protected modules — `src/thermal_solver.py`, `src/composite_materials.py`
and `src/exhaustive.py` — carry the numerical core; change them only with a test
that fails first.
