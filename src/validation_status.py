"""
validation_status.py
RISC3D-AI :: the status every reported temperature carries.

Standing rule: the label says exactly what has been compared and nothing more.

  verified          -- the equations are solved correctly. Every steady-state
                       solve checks its algebraic residual, its energy balance
                       and the convective floor, and raises if any fails.
  spreading         -- how heat spreads from the sources across the die has
    validated          been compared with measured hardware and agrees. See
                       validation/cases/ttc-samos2019-hotspot.json.
  level not         -- the absolute temperature has NOT. On that same case the
    validated          whole map sits 2.80 K below the measurement, a common
                       offset that a series resistance outside this model
                       accounts for, and the boundary coefficient was fitted
                       rather than measured.

The distinction is the point. A constant offset on every sensor is what an
unmodelled contact or an assumed boundary condition produces, and one number
removes it; it says nothing about the conduction. What survives removing it is
shape, which is what this solver computes. Claiming the first would be
claiming credit for a calibration.

One definition, used by the Stage 1 results block, both Stage 1 HTML views,
the Stage 2 terminal summary and the Stage 2 report, so they cannot drift
apart. Edit LIMITATIONS as the validation plan closes them.
"""

from __future__ import annotations

import html as _html
from typing import Tuple

STATUS = "verified; spreading validated, absolute level not validated"

VERIFIED = ("every steady-state solve checks its algebraic residual, its energy "
            "balance and the convective floor, and raises if any check fails")
VALIDATED = ("the spatial temperature distribution has been compared with a measured "
             "thermal test chip (Terraneo et al., SAMOS 2019, CC-BY dataset): 16 diodes, "
             "28.95 K of measured spread reproduced to 0.63 K RMS after removing a common "
             "offset, with the model's two unpublished inputs fitted on a DIFFERENT power "
             "pattern and held fixed")
NOT_VALIDATED = ("the absolute level has not. On that case every sensor reads 2.80 K above "
                 "the model together -- the signature of a series resistance outside this "
                 "model -- and the convective coefficient was fitted, not measured, so the "
                 "level is calibrated and only the shape is evidence")
TRANSIENT_NOTE = ("this is a transient snapshot; the per-solve checks apply to the "
                  "steady state, not to time-stepped fields")

LIMITATIONS: Tuple[Tuple[str, str], ...] = (
    ("Uniform-silicon blocks",
     "each die and chiplet is homogeneous silicon, with no BEOL, device layer "
     "or interposer TSVs in the stack that ships. Effective properties for two "
     "of those now exist and are cited (src/composite_materials.py): a "
     "TSV-bearing interposer at kxy 114.4 / kz 175.2 W/(m K), which reproduces "
     "the source paper's own FEM cell to 0.8 %, and a copper/SiO2 BEOL at "
     "kxy 118 / kz 2.1. Neither is wired in: the interposer swap alone moves "
     "the edge-AI peak +2.8 K, and a 5-15 um BEOL is thinner than one 20 um "
     "z-cell, so both need the compact-layer treatment and a deliberate re-pin"),
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
    ("One measured case, and only its shape",
     "the spreading has been compared with a measured thermal test chip and agrees "
     "(0.63 K RMS against 28.95 K of spread, parameters fitted on a different power "
     "pattern); the level on that same case is 2.80 K off and the boundary coefficient "
     "was fitted, so the absolute temperature is calibrated rather than validated. One "
     "package, forced convection, uniform chiplet power -- nothing outside that has been "
     "compared with hardware (validation/MEASURED_DATA.md)"),
    ("The cooling number dominates the budget",
     "everything above the die is one convective coefficient, and perturbing it by "
     "the +/-9 % a real rig's own h varies across 13 power patterns "
     "(tools/boundary_sensitivity.py) moves the peak by about 5 % of the rise -- "
     "more than the mesh (1-2 %) and more than the conductivities. The reported "
     "number is limited by how well the cooling is known, not by the solver"),
    ("A single constant cooling coefficient",
     "measured on that same dataset, the effective coefficient implied by 13 power "
     "patterns varies 4x under natural convection and 5x with an unpowered heat sink, "
     "but only +/-9 % under forced convection. One constant h is therefore sound for "
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
        f"{indent}  Validated (spreading only): {VALIDATED}.",
        f"{indent}  Not validated (absolute level): {NOT_VALIDATED}.",
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
        f"Validated (spreading only): {e(VALIDATED)}. "
        f"Not validated (absolute level): {e(NOT_VALIDATED)}.{note}"
        f"<ol style='margin:6px 0 4px 18px;padding:0'>{items}</ol>"
        f"Materials: {e(MATERIALS_NOTE)}."
    )
