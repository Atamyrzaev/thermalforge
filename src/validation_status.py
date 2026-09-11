"""
validation_status.py
RISC3D-AI :: the status every reported temperature carries.

Standing rule of the validation plan: until the dac143 FFT benchmark (step
5) is reproduced, every reported temperature is labelled VERIFIED, NOT
VALIDATED on the report itself, alongside the model's known limitations.

  verified   -- the equations are solved correctly. Every steady-state solve
                checks its algebraic residual, its energy balance and the
                convective floor, and raises if any check fails.
  validated  -- the equations describe a real package. That needs agreement
                with an independent benchmark, and none has been run yet.

One definition, used by the Stage 1 results block, both Stage 1 HTML views,
the Stage 2 terminal summary and the Stage 2 report, so they cannot drift
apart. Edit LIMITATIONS as the validation plan closes them.
"""

from __future__ import annotations

import html as _html
from typing import Tuple

STATUS = "verified, not validated"

VERIFIED = ("every steady-state solve checks its algebraic residual, its energy "
            "balance and the convective floor, and raises if any check fails")
NOT_VALIDATED = ("no result has been compared with an independent benchmark yet "
                 "(validation plan step 5: dac143 FFT)")
TRANSIENT_NOTE = ("this is a transient snapshot; the per-solve checks apply to the "
                  "steady state, not to time-stepped fields")

LIMITATIONS: Tuple[Tuple[str, str], ...] = (
    ("Uniform-silicon blocks",
     "each die and chiplet is homogeneous silicon, with no BEOL, device layer "
     "or interposer TSVs"),
    ("Volumetric power",
     "power fills the whole die thickness instead of entering at the active plane"),
    ("Flip-chip orientation, partly fixed",
     "the sink has been on the die backside since validation step 1, but the "
     "active plane is still not placed at the bumped face (see volumetric power)"),
    ("Unconverged mesh",
     "no grid-convergence (GCI) study at the default mesh yet, so the "
     "discretization error is unquantified"),
    ("No independent benchmark",
     "Validation step 5 requires the dac143 FFT benchmark source, which is not present. "
     "Status remains VERIFIED, NOT VALIDATED"),
)

MATERIALS_NOTE = ("conductivities from Rao et al. 2023, doi:10.2298/TSCI220805061R; "
                  "bump pitch assumed 40 um; rho and cp uncited (they affect "
                  "transients only)")


def text_block(indent: str = "  ", transient: bool = False) -> str:
    """The status and limitations as terminal lines."""
    lines = [
        f"{indent}STATUS: {STATUS.upper()}.",
        f"{indent}  Verified: {VERIFIED}.",
        f"{indent}  Not validated: {NOT_VALIDATED}.",
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
        f"Not validated: {e(NOT_VALIDATED)}.{note}"
        f"<ol style='margin:6px 0 4px 18px;padding:0'>{items}</ol>"
        f"Materials: {e(MATERIALS_NOTE)}."
    )
