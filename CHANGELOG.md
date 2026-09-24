# Changelog

Dates are when the work landed. Every number quoted here was measured on this
repository, not estimated.

## 0.9.1 — 24 September 2026

**The copilot reads plain English now, and the number that says so was
measured on sentences nothing was fitted to.** It used to read a grammar: a
list of phrasings somebody thought of, and "not understood" for the rest. It
now reads a trained model (`ai/language.py`), and the grammar stays as the
fallback for when nothing is registered.

**How it generalises.** A clause has its part names and its numbers masked --
"keep <part> under <temp>" -- before anything is featurised, so the model
learns the shape of a request rather than one design's vocabulary, and works
on parts it has never seen. Features are hashed word 1-2 grams and character
3-5 grams; the characters are what let "memroy", "srdes" and "under 90 degres"
land near the right answer. It is a softmax over 19 intents, 653 KB of
weights, offline, and it trains in under two minutes.

**What it is measured on.** Three sets of requests written by hand, in a
different voice from the generator, including typos, domain slang and things
the app cannot do:

| held-out set | the grammar | the trained reader |
|---|---|---|
| A (101 requests) | 61 % exact, F1 0.78 | **96 % exact, F1 0.98** |
| B (47, written after A's lessons) | 38 %, F1 0.51 | **87 %, F1 0.94** |
| C (35, written last, measured once) | 46 %, F1 0.66 | **80 %, F1 0.89** |

Set C is the honest headline. A and B were each measured several times while
the parser was fixed, and a set you measure repeatedly stops being held out,
however carefully you avoid copying its sentences into training — which is why
C exists and why its number is the lower one.

**It abstains instead of guessing.** Precision is 0.89-0.99 across the three
sets: when it acts, it is almost always right, and when it is unsure it says
which words it did not understand rather than applying the nearest keyword
match. The abstention threshold is chosen on generated requests run through
the real pipeline, under a constraint that it must leave 95 % of generated
junk requests alone; it is never tuned on the hand-written sets.

**Four fixes that came from watching it fail**, each a mechanism rather than a
phrasing:
- **"and" is a weak boundary.** "keep the cpu and gpu in the middle" is one
  constraint over two parts; "the dram on the left and nothing over 120
  degrees" is two asks. A quantity settles it -- a number belongs to exactly
  one ask -- and without one, splitting has to beat the whole-phrase reading
  by a margin, because "keep the cpu" reads as a real instruction on its own.
- **Pronouns.** "put the cpu next to it" resolves "it" to the part named just
  before.
- **Foreign units.** "increase the bus width to 512 bits" masks to `<other>`,
  not to a bare number, which is what stopped it reading like a temperature.
- **Typos in units.** "90 degres" and "30 wats" are read as quantities; a unit
  the reader drops takes the whole constraint with it.

**What did not change: the guarantee.** The model decides what to ASK the
solver. Every temperature in an answer is still a citation of a solve record,
the narrative builder still refuses prose containing a temperature, and the
audit still runs before anything is shown. A misread request can now answer a
question nobody asked -- visibly, with the constraints it applied written out
-- but it cannot invent a number.

## 0.9.0 — 23 September 2026

**The AI layer, under one rule: AI proposes, the solver decides.** Nothing in
it produces a number a user is shown. It ranks, screens, orders and explains;
every temperature still comes from a solve that passed its checks. The rule is
not a convention here, it is enforced in code, and the S4 gate is met on all
three counts.

**Solve records, and text that cannot outrun them** (`ai/records.py`). Every
solve the layer touches becomes a record: inputs, outputs, checks, the commit,
and an id that is a hash of all of it. A temperature reaches text only as a
citation of a record, through two independent walls. The narrative builder has
no method that accepts a bare temperature — prose containing one is refused —
and the finished text is audited against the store before it is shown, so a
number typed into a template is caught as surely as one a model invented.
Arithmetic stays traceable too: "with 7 K to spare" is a derived record naming
its parents. Tests include an explainer that states an uncited temperature and
one that cites a real record but prints the wrong number; both are refused and
the deterministic template is used instead.

**A versioned data pipeline** (`ai/dataset.py`, `tools/ai_dataset.py`). A
160-package, 3,200-label set made by the verified solver: 1,912 designs to
train and calibrate on, and 506 held-out ones across 32 packages. Each label
carries its residual, energy balance, band, mesh and commit; the manifest
carries the generator's seed and the labels' sha256, and loading a dataset
whose bytes no longer match is refused. The split is by PACKAGE, never by
layout. Generation is parallel and seeded per package, so three workers give
the same dataset as one — a test holds them equal.

**A surrogate that is physics first** (`ai/rom.py`, `ai/surrogate.py`). A
reduced-order model collapses the package to one node per material layer, with
the engine's own materials, cooling layers and boundary conditions, and solves
that in milliseconds. On its own it lands within a few per cent — that is what
carries to packages it has never seen. What a coarse cell cannot resolve, a
small numpy ensemble learns as a per-part correction, from features that
describe exactly that: how concentrated a part's power is, how small its
smallest block is against the cell, how it compares with the spreading length.
On 506 designs across 32 packages it never
trained or calibrated on, the 95th-percentile error is
**0.36 of each design's own error band** (the gate
asks for 1), inside the band on 100 % of designs, at
28 ms a design. The reduced-order model alone scores
1.21, which is the honest measure of what the
learning added.

**Uncertainty that was corrected when it undercovered.** The interval is
split-conformal, and calibrated per DESIGN it covered 92 % against a nominal
95 %: designs on one package are not exchangeable with designs on another,
because they share its cooling, its stack and its mesh. Calibrating by package
fixed it — measured coverage 100 %.

**Guided exact search** (`ai/guided.py`). The bound is untouched, so the answer
is untouched: on every instance of the regression set the optimum is
**identical**. What changed is the leaves. A complete layout is first checked
on a few prior cells — the running hot cell, the new part's own hotspot, and
the cells that have bound the peak most often so far, which the search learns
as it goes. A layout that already reaches the incumbent there cannot win and is
cut without being scored. Layouts scored fall from 80,185
to 874 over the set: **90x in
total, 52x on the median instance**. Of that, the
exact screen is 85x and the learned prior a further
1.1x.

**And what was measured and dropped.** Learned branch ORDERING — the obvious
thing to build — makes the metric worse. Dynamic rules cut interior nodes but
leave the flexible parts for the leaf level, which is where layouts are
actually scored: fail-first scored up to 150x MORE. Handing the search the true
optimum as its starting incumbent, the best any ordering could do, saves at
most 4.7x. That is why the gain is at the leaves and not in the order, and the
policy machinery stays in the module measured rather than shipped as a default.

**The app runs the guided search now.** It is exact, it returns the same
optimum, and over the regression set it takes 24 % less wall time than the
threaded search it replaces — single-threaded, because scoring far fewer
layouts beats spreading the scoring over cores. Pause and progress still work:
both are on a clock rather than a node count, so a pause takes effect within
0.2 s. On S3's own gate the worst proof of 8 distinct chiplets falls from
**17.2 s to 7.5 s**. The gate's count does not move: 10 parts now finishes in
90 s instead of running past 120, which is faster and still over the minute.

**Say it in words** (`ai/copilot.py`). "under 85 C at 30 W, memory on the left,
SerDes at the edge" becomes a limit, a power scaling and two placement
constraints; the constraints filter candidate positions, so what comes back is
a proven optimum over the constrained set, and "CPU next to memory" becomes an
exact cap on interconnect length. If the limit is missed, it climbs the cooling
ladder one exact run at a time. It is a grammar, not a language model:
deterministic, offline, and it reports the words it did not understand instead
of guessing. A language model can sit in front of it later; its text goes
through the same audit.

**Propose the package** (`ai/inverse.py`). Parts, a limit and a netlist in; a
package size, a cooling stack and an interposer out, with a Pareto front over
peak temperature, area and interconnect length where every point is a proven
optimum with its record. The surrogate screens the whole grid of candidates in
milliseconds and the exact optimizer runs only on the survivors — a screen can
cost the optimality of the PROPOSAL, never the truth of a reported number. An
epsilon-constraint sweep adds exact points that trade temperature for wire, and
a cap nothing can meet is reported as infeasible rather than approximated.

**Learn the unknowns from measurements** (`ai/calibration.py`). A package
family's undisclosed numbers — the cooling coefficient, an interface
resistance, a conductivity correction — fitted to measured parts by
Levenberg-Marquardt with the verified engine as the forward model, with
uncertainty from the local curvature and a k-fold check that judges the fit on
patterns it did not see. It also reports what the data does NOT determine: a
pair correlated beyond 0.95 is named as inseparable rather than printed as two
confident numbers. With sixteen sensors the SAMOS-style rig does separate h
from its TIM; with one sensor it does not, and the learner says so.

**In the app.** A Copilot panel whose every temperature is a link to the solve
record behind it, and "Where is it cooler?" on the selected part: the surrogate
scores every free position on the grid in about a tenth of a second and paints
them over the floorplan, with the coolest outlined and labelled an estimate.
Moving the part runs the real solver, as always. New endpoints: `/api/copilot`,
`/api/surrogate/what-if`, `/api/inverse`, `/api/records/<id>`, `/api/ai/status`
— all offline, all exercised by the driver's smoke run.

11 regression instances, 506 held-out
designs, 441 tests.

## 0.8.0 — 22 September 2026

**S1's gate is met: level and shape, on one test chip, under forced
convection.** The label every temperature carries now reads *verified;
validated (one test chip, forced convection) for level and shape, not
validated beyond that scope*. It used to say "spreading validated, absolute
level not validated". Three things changed it, and one of them was a defect in
the data.

**A defect in the measured data, found before it was mistaken for physics.**
Two of the SAMOS dataset's 39 recordings — `four` and `checkboard`, with the
fan — have their 16 diode channels rotated by one place. As recorded, their
power and temperature maps correlate at 0.24 and −0.43: in `four` the powered
heaters are cells 0, 1, 4, 5 and the hottest diodes are 1, 2, 5, 6. The same
patterns line up under the other two cooling conditions. Rotating back restores
0.95, and `checkboard` becomes a clean checkerboard. The check is model-free, so
the model cannot have tuned it. `tools/ttc_extract.py` rebuilds the committed
extract from the raw traces, corrects a recording only when the case is
decisive, and writes both correlations in. Before this, those two read as a
6–7 K conduction error, and `validation/MEASURED_DATA.md` used to report
them as one. The extract's description was also wrong about the sample rate: the
traces are 10 Hz, not 1 kHz.

**A fit where each unknown does one job.** The old fit chose silicon thickness
and `h` both for shape, and `h` ran to the edge of its range at 60,000. Now
`h` is solved so the modelled mean matches the measured one on `one`, and
thickness is chosen for shape. The declared case (`hotspot_2`, never fitted)
puts **16 of 16 sensors inside the stated band**: bias +1.33 K, shape
0.64 K RMS against 28.95 K of spread.

**All thirteen patterns, and a band that is confirmed rather than merely met**
(`tools/ttc_crossvalidate.py`). No pattern is fitted on itself.

- Shape is inside the band on **11 of 12**.
- Level is inside on **12 of 13**, each pattern predicted from the `h` fitted
  on the other twelve.
- The budget's boundary term predicts 5.5 % of the rise. It was set before
  this comparison and never tuned on it. The measured level scatter is 5.3 %
  RMS, mean +0.0 %.
- The one miss, `nine`, has one cause measured without the model. Under
  uniform power the rig's row means differ by 5.8 K with the fan and under
  1.6 K without it, so the fan cools the sink unevenly.

**A wrong explanation, retired by measuring it.** Calibrated on `one` alone,
the level lands on 6 of 12, and the misses grow with power. "Effective `h`
rises with power" was the obvious story. But the `h` each pattern implies does
not move with power (correlation −0.03). The real cause is conditioning: `one`
has the smallest rise, so a 1 K reference error is 12 % in `h`, and that is
multiplied eightfold at 47 W. The same work corrected a quieter overstatement.
The "±9 %" spread in `h` holds for `h` applied directly to the die. Through a
lid model the same runs imply ±23 %. What both describe is one scatter in the
rise, and that is what the band matches.

**BEOL, closed the way the TSV interposer was.** `beol_um` on the package,
and a BEOL selector in the app. The copper/SiO₂ stack (30 % Cu: kxy 118, kz
2.1 W/mK) is folded into the bump layer, keeping k·t along it and t/k through
it, because at 5–15 µm it is thinner than one z-cell. It is off by default, so
no pinned number moves. Measured across the 14 designs `tools/band_check.py`
audits (`tools/beol_sensitivity.py`), it moves the peak **−2.6 % to +1.7 % of
the rise**. It mostly cools: 118 W/mK sideways at the source outweighs 2.1 W/mK
through it. That is inside every design's combined band, but up to 1.3× the
mesh term, so it is worth setting when the stack is known. The engine and the
study call the same compaction function, and a test holds them equal.

**S3's gate, set on honest layouts** (`tools/s3_gate.py`). The decision was
to keep "proven optimal" and set the gate at what the exact search reliably
does. The scaling study said 12 parts fit in a minute, but its parts were
identical squares with seven powers. Interchangeable parts collapse by
symmetry, and 12 of them hide five pairs. With every part distinct, six random
layouts per count, judged by the worst:

- **8 pass**: worst proof 17.2 s, drag 17 ms at p95.
- **10 do not**: the first trial was still searching at the 120 s cap.

A trial that overruns is now cancelled and recorded as a failure with its
time. The gate states its mesh because search cost grows with it. At Fine the
same 8 take 27–65 s, one ran past 120 s, and a drag costs 69–90 ms, so Fine
is for checking a result, not for dragging. `tools/search_scaling.py` now
says its counts are the favourable case.

**S4 is ready to start, with its starting numbers measured**
(`tools/s4_readiness.py`).

- **The bar.** S4's gate is written in S1's units, so it is measured in them.
  Each design's own band is 6.1 / 6.5 / 6.7 % of the rise (p5 / median /
  p95) across 384 random designs: 10–40 mm packages, four cooling stacks,
  1–8 parts, every power pattern.
- **The labels.** The solver makes 3.3 clean labels a second, so 100,000 take
  about 8.4 h on this laptop. Each label carries its residual, energy balance,
  band, mesh and commit (schema v0). The split is by package.
- **The finding that shapes the pipeline.** At the standard mesh, 42 % of
  random designs put a hotspot finer than the cell. Every one is a patterned
  part on a package of 20 mm or more. A surrogate trained on those would learn
  the mesh's error. So flagged labels are re-solved at Fine, and the 12 % still
  unresolved are marked, never trained on.

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

**Power maps from an image.** Import map… now takes a PNG or JPEG as well as
CSV. A greyscale image is read by brightness — black is no power, white the
most — and averaged down to at most 64 tiles a side, far finer than any mesh,
averaging rather than sampling so a hotspot thinner than a tile is not lost
between samples. Image row 0 is the top, the same convention as a CSV, so no
flip is needed.

A **colour** heatmap is refused, with the reason. Brightness does not track
power in a colour scale — in jet or turbo, yellow is brighter than red — so
reading luminance would put the hotspot in the wrong place with no error.
Verified in the app: a red-on-blue heatmap is refused ("mean saturation 99 %")
and the part it was aimed at is left untouched.

**The hotspot guard stopped crying wolf on imported maps.** Yesterday's guard
took one tile as an imported map's smallest feature. A tile is the finest thing
a file can express, not the size of what it does express: a 40 × 40 image of a
5 mm part has 0.13 mm tiles, so every imported image was being called a lower
bound. A guard that fires on everything is one people learn to ignore. It now
measures the width of the hot region — the contiguous run at or above half the
map's range through the hottest tile. In the app, a 10-pixel hotspot now reads
resolved at 1.25 mm and 2.5 cells, while a single bright pixel is still caught
at 0.125 mm, and a uniform map does not trigger it at all.

**The interposer's material is a choice.** Plain silicon, or silicon
carrying copper TSVs (kxy 114.4, kz 175.2 W/mK — the cited composite that
reproduces its source paper's FEM cell to 0.8 %). This closes the S1 item left
open yesterday, and closes it the way that item needed: per design, not as a
new global default. Not every package has TSVs, and the one measured
validation case is a bare test chip with no interposer at all, which a global
switch would have silently misdescribed. Plain silicon stays the default, so no
pinned number moves. Choosing TSVs moves the edge-AI peak +2.8 K, verified in
the running app (88.559 °C against 85.764): the composite conducts better
through the interposer and worse along it, and with the sink on the die
backside the interposer is a sideways spreading path, so the in-plane loss
outweighs the through-plane gain.

**Parts can be fixed in place.** A Fixed checkbox in the editor, and
`"fixed": true` on a chiplet. An I/O die against the edge its bumps face, a
memory stack under a lid cut-out: parts whose position is a constraint rather
than a choice. In the exact search a fixed part is simply a chiplet with one
candidate, so the proof stays a proof — it is the best layout of everything
else around what is held. A fixed part is never turned, and never merged with
an identical free twin by the symmetry cut.

Doing it found a real defect. Single-chiplet solves were shared between parts
of the same footprint, keyed on size and internal pattern, on the assumption
that such parts had the same set of candidate positions. A fixed part broke
that: it received its free twin's field, with hundreds of rows where it had
one. The key now carries the positions too; identical free parts still share
(the RISC-V scenario still costs 452 unit solves, same optimum).

Verified in the running app: pin the edge-AI HBM and optimize, and it stays
at (0, 0) while the other three move; the proof covers 331,776 layouts
instead of 8,294,400 — exactly 25× fewer, the number of positions the HBM
had — and reaches the same 73.041 °C optimum, because its start was already
in the optimal layout.

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
