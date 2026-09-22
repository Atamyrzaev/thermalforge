# Measured data: what has been found, and what it can prove

The project's standing limitation was that no result had been compared with a
measured package. This file records the search for that input, the two real
cases found, and exactly what each can and cannot prove — so nobody mistakes a
case that *calibrates* the model for one that *validates* it.

## What makes a case able to validate anything

`tools/correlate.py` splits the disagreement between model and measurement in
two, because the halves mean different things:

- **A common offset on every sensor** is what an unmodelled series resistance
  looks like — a contact, a grease line, a heat sink whose real performance
  was never disclosed. One number removes it. It says nothing about whether
  the conduction inside the package is right.
- **What is left after removing that offset** is shape: how heat spreads from
  the sources across the die. That is what this solver computes, and the only
  part a case can confirm.

So a case earns a verdict by **varying in space**. The harness refuses to call
a case validating when it cannot discriminate — a single measurement, or
several that agree with each other to within their own uncertainty — because
every model that gets the level right agrees with such a case, including a
wrong one.

## Candidate 1 — A\*STAR 2.5-D package on a through-silicon interposer

`cases/astar-tsi-2p5d-bare-die.json`

Zhang et al., *Thermal Characterization of Both Bare Die and Overmolded 2.5-D
Packages on Through Silicon Interposers*, IEEE T-CPMT 2014.
Open copy: <https://oar.a-star.edu.sg/storage/6/6d1pjjxg5x/pub13-301-ipp-038-pp.pdf>

Real, independently measured, and geometrically close to what this app models:
a 5.08 × 5.08 mm thermal test chip on an 18 × 18 × 0.1 mm silicon interposer,
θ_JC = **0.83 K/W** junction-to-inlet-water at 8–20 W.

**It cannot validate this model, and the harness says so.** The measurement is
one lumped number, and decomposed with the paper's own figures it is dominated
by resistances this model does not compute:

| term | value | share of the 0.83 K/W |
|---|---|---|
| thermal grease (0.098 K·cm² / W over 0.258 cm²) | 0.380 K/W | 46 % |
| minichannel heat sink (extracted by the authors' own simulation) | 0.076–0.31 K/W | 9–37 % |
| silicon conduction — the part this solver computes | ≈0.078 K/W | ≈9 % |

Agreement would be bought by choosing the convective coefficient, not earned.
Run as-is, the model reads 3.22 K low at 10 W, which is close to the 3.80 K
the unmodelled grease resistance accounts for — a consistency check, not a
validation.

## Candidate 2 — SAMOS 2019 open thermal test chip dataset

Terraneo, Leva, Fornaciari, *An Open-hardware Platform for MPSoC Thermal
Modeling*, SAMOS 2019. Dataset: <https://doi.org/10.5281/zenodo.2871796>
(CC-BY).

This one **can** discriminate. A thermal test chip with a 4 × 4 array of
heating elements and 16 diodes gives, for each run, a measured power map and a
measured 16-point temperature map — spreads up to 35 K, far beyond any sensor
uncertainty. Thirteen power patterns (uniform, lines, a checkerboard, single
hot spots) under three cooling conditions, plus measured ambient.

Geometry not disclosed in the dataset, but pinned by cross-reference: the
3D-ICE 3.0 paper describes the uniform preheat on this platform as 5 W/cm²,
and the dataset's uniform baseline is 0.2 W per element, so each element is
0.04 cm² — **2 × 2 mm**, and the array is **8 × 8 mm**. Silicon thickness and
the convective coefficient remain unpublished, so they are fitted once, on a
single power pattern, and held fixed while the others are predicted.

**A finding from the data itself.** The effective heat-transfer coefficient
implied by each run, with `h` applied directly to the die, varies wildly under
natural convection and barely at all under forced convection:

| condition | implied h across the 13 patterns |
|---|---|
| bare chip | 4,760 → 18,940 W/m²K (4×) |
| heat sink, no fan | 2,190 → 10,850 W/m²K (5×) |
| heat sink **with fan** | 10,180 → 12,180 W/m²K (±9 %) |

This model takes a single constant `h`. That assumption is sound for forced
convection and structurally wrong for natural convection, where `h` depends on
the temperature it is supposed to produce. The forced-convection runs are
therefore the honest test, and the others are recorded as a known limitation
rather than fitted.

The ±9 % depends on the model used to imply it. Through the `lid_heatsink`
model the result below uses, the same runs imply 23,387 → 37,644 W/m²K
(±23 %), because a lid carries part of the resistance itself and `h` has to
move further to move the rise as much. What both describe is one scatter in
the *rise*, and that is the number the error budget has to match — which it
does (below).

## A defect in the data, found before it was mistaken for physics

Two of the 39 recordings — `heatsink_with_fan/four` and
`heatsink_with_fan/checkboard` — have their 16 diode channels **rotated by one
place**. As recorded, their power and temperature maps correlate at 0.24 and
−0.43: in `four` the powered heaters are cells 0, 1, 4, 5 and the hottest
diodes are 1, 2, 5, 6. The same two patterns line up under the other two
cooling conditions, and every other recording is best as recorded. Rotating
the channels back by one restores 0.95 — `checkboard` becomes a clean
checkerboard, every powered cell 63.8–66.7 °C and every unpowered one
55.5–58.6 °C — and the channel that wraps around lands on an unpowered corner.

The check is model-free, so the model cannot have tuned it:
`tools/ttc_extract.py` rebuilds the committed extract from the raw traces and
corrects a recording only when it is decisive (below 0.5 as recorded, above
0.9 corrected), writing both correlations into the extract. A checkerboard is
unchanged by moving two rows, so a rotation that merely ties (0.980 against
0.978 on the no-fan run) is left alone. Before this, the two looked like a
6–7 K conduction error, and the earlier write-up here reported them as one.

## Result on candidate 2

Silicon thickness and `h` were never published, so they are calibrated — each
doing one job — on the `one` pattern: `h` so the modelled *mean* matches the
measured mean, thickness for the best *shape* (100 µm; weakly constrained,
shape RMS 0.64–0.82 K across 100–560 µm). An earlier fit chose both for shape
and let `h` run to the edge of its range at 60,000 W/m²K; that fit is
superseded, and with it the "2.80 K common offset" this file used to report.

**The declared case**, `cases/ttc-samos2019-hotspot.json` — forced
convection, a single 7 W hot spot among fifteen 0.2 W cells, never fitted:

```
16 of 16 sensors inside the combined band   bias +1.33 K   RMS 1.48 K   worst 2.16 K
after removing the +1.33 K common offset:  shape RMS 0.64 K, worst 1.75 K
spread across the sensors: measured 28.95 K, modelled 27.82 K  (0.96x)
```

**Across the rig** (`tools/ttc_crossvalidate.py`, all thirteen patterns, none
fitted on the pattern it predicts):

| | inside the stated band |
|---|---|
| shape, `h` and thickness from `one` | **11 of 12** |
| level, `h` from `one` alone | 6 of 12 |
| level, each pattern from the `h` that fits the other twelve | **12 of 13** |

Calibrating `h` on `one` alone is the worst-conditioned choice the rig offers:
it has the smallest rise (9 K), so a 1 K error in the room reference is ~12 %
in `h`, and extrapolating that to 47 W multiplies it eightfold. That is exactly
the signature in the 6-of-12 row — the level *error* correlates with power
(+0.70) while the `h` each pattern implies does not (−0.03). `one` and
`hotspot_2` heat the *same cell* and imply 23,387 and 29,124 W/m²K: that is
this rig's test-to-test repeatability, not conduction. Leave-one-out spreads
the calibration across twelve tests instead of resting it on the weakest one,
and the `h` it fits is stable to ±2.4 %.

**The band is confirmed, not just met.** The budget's boundary term — `h`
±9 %, set before this comparison and never tuned on it — predicts 5.5 % of the
rise. The measured leave-one-out level scatter is **5.3 % RMS of the rise,
mean +0.0 %**. A band that happened to be wide would pass too; one whose size
matches the measured scatter is evidence that it means what it says.

**The one miss is one cause.** `nine` fails on both counts (level −3.44 K
against a 3.00 K band, shape 4.25 K against 3.11 K). Under uniform power the
rig's row means differ by 5.8 K with the fan running and by under 1.6 K without
it — measured, no model involved — so this fan cools the sink unevenly, and
`nine`, the pattern that heats the most of the unevenly cooled region, is where
one uniform `h` shows it. That is a boundary the model does not represent,
recorded as such rather than fitted.

## What this changed

`src/validation_status.py` now says **"verified; validated (one test chip,
forced convection) for level and shape, not validated beyond that scope"**.
It used to say "spreading validated, absolute level not validated"; the level
changed because it is now inside the band on 12 of 13 held-out patterns with
the band's size confirmed, not because a number was fitted to it.

## What would earn the rest

- A second, independently characterised package — ideally 2.5-D, with an
  interposer — so the evidence is not all one rig.
- Beyond forced convection: a temperature-dependent `h`, or an honest refusal
  to model natural convection with a constant one.
- Non-uniform cooling: a spatially varying `h`, which `nine` says this rig
  needs and the model does not have.
- Power structure finer than the test chip's 2 mm cells, which is where this
  tool's own mesh band is widest (2.14 % of rise).
