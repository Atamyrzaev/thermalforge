"""
tools/boundary_sensitivity.py
ThermalForge :: how much of a reported temperature is the cooling number?

The error budget already carries two measured terms: the mesh
(tools/mesh_convergence.py) and the conductivities (tools/material_sensitivity.py).
It was missing the third, and the third is the one the user is least sure of.
Everything above the die is collapsed into a single convective coefficient `h`,
and nobody knows their `h` to better than a few percent.

How wrong is it allowed to be? That is not a guess here. The SAMOS 2019 open
thermal-test-chip dataset (Terraneo et al., CC-BY; see validation/MEASURED_DATA.md)
records 13 different power patterns under one unchanged cooling rig, so the
scatter in the `h` those runs imply is a measurement of how well a single
constant `h` describes a real heat sink:

    heat sink with fan     10,180 -> 12,180 W/m2K     +/- 9 %
    heat sink, no fan       2,190 -> 10,850 W/m2K     5x
    bare chip               4,760 -> 18,940 W/m2K     4x

So +/-9 % is the honest spread for FORCED convection, and it is the figure
this study perturbs by. Natural convection is not a constant-h problem at all,
which is recorded as a limitation rather than folded into a band.

    python tools/boundary_sensitivity.py

Writes validation/boundary_sensitivity.json. Verification, not validation: it
says how much the answer moves when an input moves, not whether the answer is
right.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402

for name in ("ThermalForge.ThermalSolver", "ThermalForge.GridBuilder", "ThermalForge.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT = ROOT / "validation" / "boundary_sensitivity.json"

# Measured on the SAMOS 2019 rig: the spread in implied h across 13 power
# patterns under one unchanged forced-convection setup.
H_TOLERANCE = 0.09
SOURCE = ("spread of the effective h implied by 13 power patterns under one "
          "unchanged forced-convection rig, SAMOS 2019 dataset "
          "(doi:10.5281/zenodo.2871796); see validation/MEASURED_DATA.md")

CASES = {
    "passive": {"size_mm": 20.0, "h_w_m2k": 1000.0, "ambient_c": 25.0, "pitch_mm": 2.0},
    "forced": {"size_mm": 20.0, "h_w_m2k": 2500.0, "ambient_c": 35.0, "pitch_mm": 2.0,
               "cooling": "lid_heatsink"},
    "coldplate": {"size_mm": 20.0, "h_w_m2k": 25000.0, "ambient_c": 25.0, "pitch_mm": 2.0,
                  "cooling": "coldplate"},
}

PARTS = [
    {"id": "npu", "name": "NPU", "w_mm": 6, "h_mm": 6, "power_w": 8.0,
     "x_mm": 2, "y_mm": 2, "power_pattern": "mac_array"},
    {"id": "cpu", "name": "CPU", "w_mm": 4, "h_mm": 4, "power_w": 4.0,
     "x_mm": 14, "y_mm": 2, "power_pattern": "quad_cores"},
    {"id": "mem", "name": "MEM", "w_mm": 8, "h_mm": 6, "power_w": 2.0,
     "x_mm": 2, "y_mm": 12},
]


def peak_at(cache, raw: dict, h: float) -> float:
    spec = E.PackageSpec.parse(dict(raw, h_w_m2k=h))
    out = E.solve(cache, spec, E.parse_chiplets(PARTS, spec))
    if not out["ok"]:
        raise SystemExit(f"refused: {out['violations']}")
    return float(out["peak"]["t_c"])


def main() -> None:
    cache = E.ModelCache()
    report: Dict = {
        "what": "how far the peak moves when the convective coefficient moves by "
                "the amount a real rig's own h is known to vary",
        "h_tolerance": H_TOLERANCE,
        "tolerance_source": SOURCE,
        "not_covered": ("natural convection, where h depends on the temperature it "
                        "produces and varies 4-5x across power patterns on the same "
                        "rig -- a band around a constant h does not describe that"),
        "cases": {},
    }
    print(f"Perturbing h by +/-{H_TOLERANCE:.0%} (measured, not assumed):")
    for name, raw in CASES.items():
        t0 = time.time()
        h = float(raw["h_w_m2k"])
        base = peak_at(cache, raw, h)
        low = peak_at(cache, raw, h * (1.0 - H_TOLERANCE))    # weaker cooling: hotter
        high = peak_at(cache, raw, h * (1.0 + H_TOLERANCE))   # stronger: cooler
        rise = base - float(raw["ambient_c"])
        # Half the full swing, so it reads as a +/- like the other two terms.
        band = 0.5 * abs(low - high)
        report["cases"][name] = {
            "h_w_m2k": h,
            "cooling": raw.get("cooling", "direct"),
            "baseline_peak_c": base,
            "peak_at_low_h_c": low,
            "peak_at_high_h_c": high,
            "rise_k": rise,
            "boundary_band_k": band,
            "boundary_band_pct_of_rise": 100.0 * band / rise if rise else 0.0,
            "seconds": time.time() - t0,
        }
        print(f"  {name:10s} h={h:>8,.0f}  peak {base:7.3f} C  "
              f"({low:7.3f} .. {high:7.3f})  band +/-{band:5.3f} K  "
              f"= {100.0 * band / rise:4.2f} % of the rise")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {OUT}")
    worst = max(report["cases"].values(), key=lambda c: c["boundary_band_pct_of_rise"])
    print(f"  worst: {worst['boundary_band_pct_of_rise']:.2f} % of the rise "
          f"({worst['cooling']})")


if __name__ == "__main__":
    main()
