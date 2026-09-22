# Changelog

Dates are when the work landed. Every number quoted here was measured on this
repository, not estimated.

## 0.7.0 — 22 September 2026

**Rectangular packages.** `size_y_mm` on the package, and a Height field in
the app. Square stays the default — a package states a second side only when
it has one — so every design, scenario and test written before this still
reads unchanged. The mesh cell has to divide both sides, because a cell that
fits one and leaves a sliver on the other would mesh a package that is not the
one described. A part is bounded by the side it lies along. The drawing keeps
one scale for both axes, so a millimetre is the same length across as up and a
40 × 10 mm package looks like one instead of being stretched square.

Verified in the running app: 30 × 20 mm meshes 60 × 40 × 12, solves in 12 ms
with every check passing, and the peak drops from 85.8 to 77.9 °C against the
same parts on 20 × 20 — what 50 % more cooled area should do. The exact
optimizer proves its optimum on a rectangle the same way it does on a square.

**How far the exact proof actually scales, measured** (`tools/search_scaling.py`).
Two earlier guesses about what limits it were wrong. The mesh does not: 307,200
unknowns factorize in about 35 s with a 3.7e-12 residual. Memory does not: the
unit fields for 30 chiplets at 40 mm / 0.5 mm are 0.55 GB. What limits it is the
search. On a 40 mm package the proof stays near a second up to 8 parts, takes
12 s at 10 and 9 s at 12, and 59 s at 14. It does not grow smoothly, because
the bound bites harder when parts crowd each other. So the S3 gate's 30 chiplets
are out of reach for this search, and the reason is now a measurement.

## 0.6.0 — 21 September 2026

**The error budget has a third term, and it is the biggest one.** Everything
above the die was collapsed into one convective coefficient and never given a
band. `tools/boundary_sensitivity.py` perturbs it by ±9 % — not a guess, but
the spread the same rig's own `h` shows across 13 power patterns in the SAMOS
dataset — and the peak moves **5.3 % of the rise** passively, 5.4 % under a
lid and heatsink, 3.2 % under a cold plate. That is more than the mesh
(1–2 %) and more than the conductivities.

This corrects what the project used to say. The budget was described as
dominated by the silicon conductivity; measured, it is dominated by the
cooling number. Demo scenarios go from ±2.47, ±1.68, ±1.41 K to **±4.08,
±3.19, ±2.34 K** — wider because a term was missing, not because anything got
worse.

**The search can turn parts, and by default it does not.** 90-degree
rotations are candidate states now: a part's sides swap and everything inside
turns with them, so two engines side by side become two stacked. Squares with
symmetric insides are skipped, because turning them enumerates the identical
heat source twice. Verified against a brute force that rotates independently,
so a wrong turn would either beat the truth (an illegal layout) or miss it (a
candidate never scored); the re-solve tolerance is unchanged.

It is off by default, and that is a measurement rather than a preference. On
all three shipped scenarios turning parts moves the proven optimum **0.000 K**
while doubling the candidate count, 8,294,400 to 16,588,800. A rectangle with
the whole grid to move around in rarely needs turning as well. It earns its
cost where packing is tight — three long thin parts on a 10 mm package gain
0.156 K — so `"rotate": true` on the package is a switch for that case rather
than a tax on every search, which matters in the stage whose whole point is
speed. The certificate records whether turning was allowed and which parts
were turned, and the engine refuses to report a turned layout if it was not.

**Long proofs can be paused.** `POST /api/jobs/<id>/pause` and `/resume`, and
a Pause button on the optimizer overlay. The hold lives in the object the
engine already consults for cancellation, so it takes effect exactly where
stopping was already safe and needs no solver change. Verified on a
176,319,369,216-layout search: held after 1.3 s, progress provably frozen,
resumed to the same optimum (65.4729 °C, re-solve delta 2.0e-10 K), and still
cancellable while held.

Asking to pause is reported as **`pausing`** until the job actually stops,
because model building has no checkpoints and a request can sit unhonoured
for seconds — saying `paused` then would claim something untrue. Both UI poll
loops were also only continuing while the state was `running`, so a paused
job would have fallen through and reported a finish that never happened.

**Two more effective properties, cited.** A TSV-bearing interposer
(kxy 114.4, kz 175.2 W/(m K)), which reproduces the source paper's own FEM
cell to 0.8 %, and a copper/SiO2 BEOL stack (kxy 118, kz 2.1 — two orders of
magnitude apart, with the small number sitting on the plane where power is
deposited). Neither is in the shipped stack, and the limitation now says so
along with what it would cost: the interposer swap alone moves the edge-AI
peak **+2.8 K**, a 5–15 µm BEOL is thinner than one 20 µm z-cell, and the one
measured validation case is a bare test chip with no interposer that a global
switch would silently misdescribe.

## 0.5.1 — 20 September 2026

**"Verified, not validated" is no longer the whole truth.** The spreading has
now been compared with measured hardware. `tools/correlate.py` against the
SAMOS 2019 open thermal-test-chip dataset (Terraneo et al., CC-BY): 16 on-die
diodes, a single 7 W hot spot among fifteen 0.2 W cells under forced air.
**28.95 K of measured spatial spread reproduced to 0.63 K RMS** — and the two
inputs the source never published, silicon thickness and the convective
coefficient, were fitted on a *different* power pattern and held fixed, so
this one is a prediction.

The level was not validated and the label says so. Every sensor reads 2.80 K
above the model *together*, which is a series resistance outside this model,
and `h` was fitted rather than measured. The status is now **"verified;
spreading validated, absolute level not validated"**, stated as two separate
claims because only one of them was earned.

**The harness can tell those apart now.** `correlate.py` splits the
disagreement into a common offset (an unmodelled contact, grease line or
undisclosed heat sink — one number removes it) and shape (the conduction this
solver computes). It also refuses to call a case validating when the case
cannot discriminate: one measurement, or several agreeing within their own
uncertainty, would be matched by a wrong model too. That guard caught an
existing test which had been treating a single perfect sensor as evidence.

**Two measured findings about this model's own assumptions.** From the same
dataset, the effective `h` implied by 13 power patterns varies 4× under
natural convection and 5× with an unpowered heat sink, but only ±9 % under
forced convection — so one constant `h` is sound for forced convection and
structurally wrong for natural convection. And on the uniform pattern the
measurement varies 6.88 K across the die while the model varies 0.65 K:
uniform power under a uniform `h` with adiabatic sides gives a flat field, and
the residual is the fan, which no amount of conduction modelling will produce.

**A second real case is recorded and deliberately not used.** The A*STAR 2.5-D
interposer measurement (θ_JC = 0.83 K/W) is real and well documented, but
decomposed with the paper's own figures it is 55–83 % grease and heat sink
against ≈9 % silicon conduction. Agreement would be bought by choosing `h`.
Run as-is the model reads 3.22 K low, close to the 3.80 K the unmodelled
grease accounts for — a consistency check, not a validation. Both cases and
the reasoning are in `validation/MEASURED_DATA.md`.

## 0.5.0 — 20 September 2026

**The mesh error is measured, not extrapolated.** The convergence study now
refines to 0.25 mm — affordable since the app installs its own fill-reducing
factorization — so the error at the shipped mesh is compared against a finer
solution instead of estimated from coarser ones. On the reference package that
made the old figure look pessimistic: 0.5 mm sits 0.038 K (passive) and
0.240 K (cold plate) from the 0.25 mm answer, against published GCI estimates
of 0.43 K and 1.76 K. Observed order is clamped to the scheme's formal second
order, because an apparent 3.3 means the differences are near the noise floor,
not that the method improved.

**Then the band got wider, not narrower, because the narrow one was wrong.**
Measuring the reference package more precisely says nothing about designs
unlike it, and the band is scaled to whatever design is on screen. Two things
came out of checking that, both below: internal hotspots were changing size
with the mesh, and the reference set did not span the concentrated designs the
app permits. The band quoted for a uniform design is now 1.34 % of the rise
where it was 0.38 %, and 2.14 % with internal power patterns. Every band takes
the worst reference case in its family, and says on its face that it is an
estimate from reference packages.

**Internal hotspots stopped changing size with the mesh.** A chiplet's power
blocks are fractions of its footprint, and they were being rounded to the whole
cells they mostly covered. That kept the watts but changed the block's modelled
AREA with the mesh, so its power density — the thing that sets the peak —
changed too. Refining did not converge, it wandered: the patterned reference
package read 181.5, 139.2, 149.3 and 153.6 °C at 2, 1, 0.5 and 0.25 mm.
Rounding also misplaced a sub-cell block by a whole cell (a block at 0.45–0.47
of the footprint landed in the cell spanning 0.50–0.75). Blocks now take the
cells they geometrically overlap, so the same physical hotspot is modelled at
every mesh. The reference package now reads 120.5, 144.8, 150.1 and 152.3 °C —
converging from below, as an under-resolved hotspot should. On the shipped
RISC-V scenario the difference between 0.5 mm and 0.25 mm fell from **1.008 K
to 0.013 K**. Power is conserved to 1e-12 either way, which is why no test
caught this: nothing was lost, it was in the wrong place.

This supersedes the pattern temperatures quoted in 0.3.x and 0.4.0, which were
measured with the old deposition.

**The band knows which kind of design it is describing.** The convergence study
now runs four reference cases — uniform and patterned chiplets, passive and
under a cold plate — because the uniform cases understate a patterned design by
more than an order of magnitude (measured on the RISC-V scenario before the fix
above: 0.004 K uniform against 1.008 K patterned). A design is banded by the
family it belongs to, taking the worst case in that family.

**A hotspot smaller than a cell is now refused, not averaged silently.**
Auditing the band found a case it never covered. On a 5 mm part carrying 70 %
of 8 W in a centred block, refining 0.5 mm → 0.25 mm moves the peak +0.2 K
when the block is 3.15 mm across, +1.0 K at 1.00 mm, and **+29.2 K at
0.50 mm** — against a band of 1.98 K. And blocks of 0.50, 0.25 and 0.10 mm
report the *same* temperature, because each fills exactly one cell: below a
cell the peak describes the mesh, not the design, and refining raises it
rather than converging on it. The app now measures the smallest feature in
every design, fails a **Hotspots resolved** check when it is under two cells,
labels the peak a lower bound, and declines to quote a mesh band it knows does
not apply. It also says when no mesh it builds would resolve the feature — a
0.1 mm hotspot needs 0.05 mm cells — rather than offering the fine mesh as a
fix that is not one.

**The band is audited now, not asserted.** `tools/band_check.py` solves
fourteen designs at both mesh levels and compares the difference it measures
against the band the app quoted them. Run against the reference set as it
stood, four exceeded their band — the worst by 2.85× — and every one was a
uniform design whose heat came from one small part, which the reference cases
(4 to 10 mm chiplets) did not represent. Two concentrated reference cases were
added and all fourteen now come in under, worst ratio 0.81 on the design that
used to break it. The band is 1.34 % of the rise for uniform chiplets and
2.14 % with internal patterns, against 0.38 % before — wider because the
narrow one was wrong, which is the only good reason to widen a band.

**Measure this design's mesh error.** A button, and `POST /api/mesh-check`:
solve the design on screen at both mesh levels and subtract. That is a
measurement of this design rather than an estimate scaled from a reference
package, and it is what to quote when a design sits near its limit. Measured on
the three demo scenarios: 0.062, 0.046 and 0.013 K, against estimates of 1.30,
1.09 and 0.75 K.

**A finer mesh is a choice now.** Each level takes the finest cell that fits
its unknown budget: Standard 36,000, Fine 120,000. On a 20 mm package that is
0.5 mm against 0.25 mm — under a second against 4–25 s, depending on the
cooling stack. Exact at either size — residuals near 1e-11 — so it costs time
and memory, not correctness. The model cache evicts by size as well as count,
because a fine model carries roughly 640 MB of factorization.

Small packages gain for free: a 10 mm package now meshes at 0.25 mm inside the
same 36,000-unknown standard budget, which also halves how much the assembly
keep-out is over-enforced — 250 µm applied for a 100 µm rule, where it was
500 µm.

Measured budgets after all of it: edge-AI **85.76 ± 2.47 K**, cold plate
**75.71 ± 1.68 K**, RISC-V **69.76 ± 1.41 K** — about 4 % of the rise. The
mesh half of that is a conservative estimate; measured on these three designs
it is 0.062, 0.046 and 0.013 K, and the button that measures it ships with
them.

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

**Fixed: a leak the burst left behind.** Each burst installed a second
factorization of a package-sized system into a cached, shared solver and
never released it — 20 to 80 MB per run, 563 MB after four cooling paths.
It showed up as a test suite that was fast per file (51 s, 96 s, 11 s) and
took 43 minutes for the three together, through paging. Released at the end
of the run: the suite is back to 162 s and repeated bursts hold flat.

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
