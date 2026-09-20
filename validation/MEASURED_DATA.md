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
implied by each run varies wildly under natural convection and barely at all
under forced convection:

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

## Result on candidate 2

`cases/ttc-samos2019-hotspot.json` — forced convection, a single 7 W hot spot
among fifteen 0.2 W cells. Silicon thickness and `h` were fitted on the `one`
pattern and held fixed; this pattern was never fitted.

```
16 of 16 sensors outside the combined band   bias -2.80 K   RMS 2.87 K
after removing the -2.80 K common offset:    shape RMS 0.63 K, worst 1.81 K
spread across the sensors: measured 28.95 K, modelled 27.72 K  (0.96x)
```

**The shape agrees; the level does not.** 28.95 K of measured spatial variation
is reproduced to 0.63 K RMS — about 2 % of the signal — while the whole map
sits 2.80 K low together. A uniform offset is what a series resistance outside
the model produces, and the convective coefficient was fitted rather than
measured, so the level is calibrated and carries no evidence.

Held out across all twelve non-fitted patterns, shape RMS averages 2.29 K.
It is best where the heat is concentrated (hot spots 0.63–0.99 K, lines
0.38–1.01 K) and worst where it is spread out (`four` 7.28 K, `checkboard`
5.92 K). The uniform `sixteen` pattern is the clearest statement of the
limitation: the measurement varies by 6.88 K across the die and the model by
0.65 K, because uniform power under a uniform `h` with adiabatic sides gives a
nearly flat field. That residual is non-uniform cooling — the fan — and no
amount of conduction modelling will produce it.

## What this changed

`src/validation_status.py` no longer says "verified, not validated". It says
**"verified; spreading validated, absolute level not validated"**, and states
both halves separately, because they are different claims and only one of them
has been earned.

## What would earn the rest

- The level: a case where the boundary condition is measured rather than
  fitted, or where the interface resistance is disclosed.
- Beyond forced convection: either a temperature-dependent `h`, or an honest
  refusal to model natural convection with a constant one.
- Beyond uniform chiplet power: a measured case with internal power patterns,
  which is where this tool's own error band is widest (2.14 % of rise).
