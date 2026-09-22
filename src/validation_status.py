"""
validation_status.py
RISC3D-AI :: the status every reported temperature carries.

Standing rule: the label says exactly what has been compared and nothing more.

  verified          -- the equations are solved correctly. Every steady-state
                       solve checks its algebraic residual, its energy balance
                       and the convective floor, and raises if any fails.
  validated on one  -- the temperature map of a measured thermal test chip
    test chip,         under forced convection, predicted with nothing fitted
    forced             on the pattern predicted, lands inside the model's own
    convection         stated band: level and shape both, on the declared case
                       (validation/cases/ttc-samos2019-hotspot.json) and on
                       all but one of the rig's thirteen patterns
                       (tools/ttc_crossvalidate.py).
  not validated     -- everything outside that scope: natural convection,
    beyond it          non-uniform cooling, stacked and 2.5-D packages,
                       transients, power structure finer than 2 mm.

Two claims are kept apart because they fail differently. A common offset on
every sensor is what an unmodelled contact or a wrong cooling number produces;
what survives removing it is shape, which is the conduction this solver
computes. Both are now inside the band on that rig -- and the band's own
boundary term, never tuned on this data, predicts the level scatter it
measures (5.5 % of the rise predicted, 5.3 % RMS measured). The scope is still
one chip and one cooling regime, and the label says so.

One definition, used by the Stage 1 results block, both Stage 1 HTML views,
the Stage 2 terminal summary and the Stage 2 report, so they cannot drift
apart. Edit LIMITATIONS as the validation plan closes them.
"""

from __future__ import annotations

import html as _html
from typing import Tuple

STATUS = ("verified; validated (one test chip, forced convection) for level and shape, "
          "not validated beyond that scope")

VERIFIED = ("every steady-state solve checks its algebraic residual, its energy "
            "balance and the convective floor, and raises if any check fails")
VALIDATED = ("the temperature map of a measured thermal test chip (Terraneo et al., SAMOS "
             "2019, CC-BY dataset: 16 diodes, 13 power patterns, forced convection), "
             "predicted with nothing fitted on the pattern being predicted. The declared "
             "case puts 16 of 16 sensors inside the stated band -- level bias +1.33 K, "
             "shape 0.64 K RMS against 28.95 K of spread. Across the rig the shape is "
             "inside the band on 11 of 12 held-out patterns and the level on 12 of 13 "
             "(each from the cooling coefficient fitted on the other twelve), and that "
             "level scatter, 5.3 % RMS of the rise, is what the budget's boundary term "
             "predicts (5.5 %)")
NOT_VALIDATED = ("natural convection, where one constant h is structurally wrong; "
                 "non-uniform cooling (the one pattern that misses, "
                 "'nine', is where this rig's fan cools the sink unevenly); stacked and "
                 "2.5-D packages and interposers, which the test chip does not have; the "
                 "BEOL option, which was not part of the comparison; "
                 "transients; and power structure finer than its 2 mm cells. And the level "
                 "is only as good as the cooling coefficient supplied: what was shown is "
                 "that one h describes a real rig to within the band, not that any h a "
                 "user types is right")
TRANSIENT_NOTE = ("this is a transient snapshot; the per-solve checks apply to the "
                  "steady state, not to time-stepped fields")

LIMITATIONS: Tuple[Tuple[str, str], ...] = (
    ("Homogeneous blocks unless a design says otherwise",
     "each die and chiplet is homogeneous silicon by default. Two cited effective "
     "media are per-design options (src/composite_materials.py), both off by "
     "default so no pinned number moves: a TSV-bearing interposer "
     "(interposer_material; kxy 114.4 / kz 175.2 W/(m K), reproducing its source "
     "paper's FEM cell to 0.8 %; +2.8 K on the edge-AI peak) and a copper/SiO2 BEOL "
     "(beol_um; kxy 118 / kz 2.1 at 30 % Cu), folded into the bump layer because "
     "at 5-15 um it is thinner than one 20 um z-cell. Across the 14 designs "
     "tools/band_check.py audits, a BEOL moves the peak -2.6 % to +1.7 % of the "
     "rise (tools/beol_sensitivity.py) -- inside every design's combined band, but "
     "up to 1.3x its mesh term, so it is worth setting when the stack is known. "
     "Mostly cooler, because it spreads heat sideways right at the source. The "
     "copper fraction is a process parameter and is fixed at 30 %"),
    ("Device layer one cell thick",
     "power now enters at the active plane against the bumps, as a flip-chip die "
     "generates it, but the source is one mesh cell (20 um) thick rather than the "
     "sub-micron real device layer; halving that thickness moves the peak by "
     "+0.002 K passive and +0.06 K under a cold plate"),
    ("Mesh error quantified, not removed",
     "a convergence study refined to 0.25 mm over six reference cases "
     "(tools/mesh_convergence.py) puts the band at 1.34 % of the rise for uniform "
     "chiplet power and 2.14 % with internal power patterns; tools/band_check.py "
     "audits that band against fourteen designs solved at both mesh levels, and the "
     "app prints it beside every peak and can measure this design's own instead"),
    ("One measured test chip, one cooling regime",
     "13 power patterns on one 8 x 8 mm test chip under one forced-convection rig "
     "(validation/MEASURED_DATA.md). The rig's silicon thickness and cooling "
     "coefficient were never published, so they are calibrated on OTHER patterns of "
     "the same rig: every compared number is a prediction, but none is independent of "
     "that rig. Two of the dataset's 39 recordings have their diode channels rotated "
     "by one place; the extract detects that without the model and corrects it "
     "(tools/ttc_extract.py), and without the correction the shape looked 6-7 K wrong "
     "on exactly those two. A second, independently characterised package -- ideally "
     "a 2.5-D one -- is what would widen this"),
    ("The cooling number dominates the budget",
     "everything above the die is one convective coefficient, and perturbing it by "
     "+/-9 % (tools/boundary_sensitivity.py) moves the peak by about 5 % of the rise "
     "-- more than the mesh (1-2 %) and more than the conductivities. That 9 % is the "
     "spread a real rig's 13 patterns imply for h applied directly to the die (the "
     "same runs imply +/-23 % through a lid model, because the lid carries part of the "
     "resistance itself); what both describe is one scatter in the rise, and measured "
     "leave-one-out on that rig it is 5.3 % RMS against the term's 5.5 %. The reported "
     "number is limited by how well the cooling is known, not by the solver"),
    ("A single constant cooling coefficient",
     "measured on that same dataset (h applied directly to the die), the effective "
     "coefficient implied by 13 power patterns varies 4x on the bare chip and 5x on a "
     "heat sink without its fan, but only +/-9 % with the fan running. One constant h is therefore sound for "
     "forced convection and structurally wrong for natural convection, where h depends "
     "on the temperature it is supposed to produce"),
)

MATERIALS_NOTE = ("conductivities from Rao et al. 2023, doi:10.2298/TSCI220805061R; "
                  "bump pitch assumed 40 um; rho and cp uncited (they affect "
                  "transients only)")


def text_block(indent: str = "  ", transient: bool = False) -> str:
    """The status and limitations as terminal lines."""
    lines = [
        f"{indent}STATUS: {STATUS.upper()}.",
        f"{indent}  Verified: {VERIFIED}.",
        f"{indent}  Validated (one test chip, forced convection): {VALIDATED}.",
        f"{indent}  Not validated (outside that scope): {NOT_VALIDATED}.",
    ]
    if transient:
        lines.append(f"{indent}  Note: {TRANSIENT_NOTE}.")
    lines.append(f"{indent}  Remaining limitations:")
    for i, (name, detail) in enumerate(LIMITATIONS, start=1):
        lines.append(f"{indent}    {i}. {name}: {detail}.")
    lines.append(f"{indent}  Materials: {MATERIALS_NOTE}.")
    return "\n".join(lines)


def html_block(transient: bool = False) -> str:
    """The status and limitations as an HTML fragment (no wrapper element,
    so each report places it in its own styled container)."""
    e = _html.escape
    items = "".join(f"<li><b>{e(name)}</b>: {e(detail)}.</li>" for name, detail in LIMITATIONS)
    note = f"<br><b>Note:</b> {e(TRANSIENT_NOTE)}." if transient else ""
    return (
        f"<b>{e(STATUS.capitalize())}.</b> Verified: {e(VERIFIED)}. "
        f"Validated (one test chip, forced convection): {e(VALIDATED)}. "
        f"Not validated (outside that scope): {e(NOT_VALIDATED)}.{note}"
        f"<ol style='margin:6px 0 4px 18px;padding:0'>{items}</ol>"
        f"Materials: {e(MATERIALS_NOTE)}."
    )
