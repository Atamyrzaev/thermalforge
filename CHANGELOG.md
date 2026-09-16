# Changelog

Dates are when the work landed. Every number quoted here was measured on this
repository, not estimated.

## 0.4.0 — 17 September 2026

**Over time.** A burst, time-stepped: idle → full power → idle, every step a
Crank–Nicolson step of the same verified solver. The implicit system is
symmetrized and factorized the way the steady one is, which keeps a lidded
package at ~30 ms per step instead of the 732 ms an iterative fallback costs.
The check the mode rests on is reported, not assumed: run long enough and the
transient arrives at the steady field (88.88 °C against 88.88). The scheme's
own step artefact is measured too (58 mK at dt = 3.9 ms, 10 mK at 2 ms,
against a 45 K rise) and shown beside the curve.

**Correlation harness.** `tools/correlate.py` consumes a case file — a package
plus the temperatures somebody measured on it — and reports agreement against
two bands, the instrument's uncertainty and the model's error budget. A
synthetic example ships so the process can be rehearsed before hardware
arrives; every output derived from it is stamped SYNTHETIC, because the
failure that would matter here is a made-up number being quoted later as a
measurement.

**Packaging.** The built folder carries a README_FIRST that explains the
unsigned-binary warning, and `--zip` packs it into one file to hand over.

## 0.3.0 — 15–16 September 2026

**Buildable layouts.** A minimum chiplet-to-chiplet clearance (default 100 µm,
for underfill flow and bump dispense tolerance) constrains validation, the
drag feedback and the search, so the proven optimum is the coolest
*manufacturable* layout. Corner-to-corner contact is refused too. Enforced in
whole mesh cells and rounded up, with the app stating what it enforced.

**Chiplets are not uniform heaters.** Internal power blocks as fractions of the
footprint, four patterns, and a CSV importer for a design team's own map —
matrix or region table. On an 8 W, 5 × 5 mm NPU the uniform assumption
under-reports the junction by 13.2 K.

**The cooling path is meshed.** TIM, lid and heatsink base as compact
anisotropic layers that preserve both the sheet conductance `k·t` and the
through-plane resistance `t/k`. A copper lid drops an 8 W hotspot by 44 K
without moving the cooling floor at all — all of it spreading.

**The search uses the machine.** Root branches across cores, batched solves
across threads (verified by substitution, since SuperLU is not documented
thread-safe), and the chiplet cap raised from 12 to 16. RISC-V end to end:
12.6 s → 9.0 s, optima unchanged to 1e-12.

**Packaged app.** `packaging/build.py` produces a folder that runs on a machine
with no Python. Its `--check` starts the built exe and solves through it, which
caught a plotting library dragged into every build and a second copy binding a
port the first one held.

## 0.2.0 — 14–15 September 2026

**Physics corrections.** Power enters at the active plane of each die rather
than filling the die thickness. The convective boundary now puts the film in
series with conduction across the half cell to the cooled surface — omitting it
under-read every temperature by 0.004 K (passive) to 0.13 K (cold plate).

**Closed-form benchmarks.** A uniformly heated slab and a rectangular flux
channel, derived in `tests/test_benchmarks.py`. The slab error is second order
with exactly the predicted `q·dz²/(8k)` coefficient; the spreading peak
converges to the Fourier series within 3 %.

**An error budget, printed.** Mesh convergence (2 / 1 / 0.5 mm) and conductivity
sensitivity, combined in quadrature and scaled to the design: ±2.1 to ±2.6 K on
the demo scenarios, dominated by the silicon conductivity rather than the mesh.

**Editable stack** and the booth-ready basics: save/open designs, undo/redo, a
warning before a slow proof, and a report you can download.

## 0.1.0 — 14 September 2026

First RISC3D Studio: interactive floorplan with live re-solve, 3-D view,
before/after comparison, and an exact placement optimizer whose winner is
re-solved from scratch and refused if the two disagree by more than 1e-6 K.
