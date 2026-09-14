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
| **Error bar** | A 2 / 1 / 0.5 mm mesh-convergence study (`tools/mesh_convergence.py`, results in `validation/`) gives the band the app prints beside the peak: ±0.4 K passive, ±1.8 K under a cold plate, on the reference package. |

Known limitations are printed on the reports themselves, not hidden here: power
enters at the active plane but is one mesh cell (20 µm) thick, silicon is
homogeneous (no BEOL, device layer or TSVs), cooling is one effective coefficient
on the die backside, and the stack is fixed at 100/40/100 µm.

## Layout

| Path | What it is |
|---|---|
| `studio/` | The app: `engine.py` (physics + exact optimizer), `server.py` (stdlib HTTP API), `static/` (the UI) |
| `src/` | The core: `thermal_solver.py`, `grid_builder.py`, `composite_materials.py`, `experiment.py`, `exhaustive.py`, `optimizer.py`, `surrogate.py` |
| `main.py` | Command-line pipeline (Stage 1 single die, Stage 2 placement) |
| `tools/mesh_convergence.py` | The mesh study behind the error bar |
| `validation/` | Measured verification artifacts, versioned as evidence |
| `tests/` | 244 tests: `test_core.py`, `test_studio.py`, `test_benchmarks.py` |

## Development

```bash
pip install -r requirements.txt
python -m pytest -q
```

The three protected modules — `src/thermal_solver.py`, `src/composite_materials.py`
and `src/exhaustive.py` — carry the numerical core; change them only with a test
that fails first.
