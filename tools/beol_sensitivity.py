"""
tools/beol_sensitivity.py
RISC3D-AI :: does leaving the BEOL out move any number the app reports?

The interconnect stack on a die's active face -- copper wiring levels in SiO2 --
is about 2 W/(m K) through and over 100 along (src/composite_materials.beol_k),
and it is 5-15 um thick, thinner than one 20 um z-cell. So it cannot simply be
given cells. It can be COMPACTED into the layer it sits against, the way the
cooling path compacts its lid and TIM: preserve what the pair conducts along
(the sum of k.t) and what it resists through (the sum of t/k), in the bump
layer's own thickness.

In a flip-chip stack the BEOL sits between the device layer, where the power
is deposited, and the bumps. Heat leaving upward through the silicon to the
sink never crosses it; heat leaving downward into the interposer does. So its
effect is on the downward, spreading path, and how much that matters is a
measurement, not an argument. This compacts BEOL across its realistic range
into every design tools/band_check.py audits and reports the shift in the peak
beside that design's own mesh band and combined band.

    python tools/beol_sensitivity.py

Writes validation/beol_sensitivity.json. Verification, not validation: it
says how much the answer moves, not which answer is right.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import src.grid_builder as G  # noqa: E402
import studio.engine as E  # noqa: E402
from src.composite_materials import beol_bump_layer, beol_k  # noqa: E402
from tools.band_check import SWEEP  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT = ROOT / "validation" / "beol_sensitivity.json"
BUMP = "bump_underfill_composite"
# (thickness um, copper fraction): the realistic range, thinnest to heaviest
VARIANTS = ((5.0, 0.30), (10.0, 0.30), (15.0, 0.30), (10.0, 0.20), (10.0, 0.40))


def compacted(base: dict, t_bump_um: float, t_beol_um: float, cu: float) -> dict:
    """The bump layer with a BEOL folded in -- the same function the engine
    uses when a package sets beol_um, so the study measures what ships."""
    return beol_bump_layer(base, t_bump_um, t_beol_um, cu)


def peak(case: dict, material: dict):
    saved = G.MATERIALS[BUMP]
    G.MATERIALS[BUMP] = material
    try:
        spec = E.PackageSpec.parse({"size_mm": case["size_mm"], "h_w_m2k": case["h"],
                                    "ambient_c": 25.0, "pitch_mm": case["pitch_mm"],
                                    "cooling": case["cooling"]})
        out = E.solve(E.ModelCache(capacity=1), spec, E.parse_chiplets(case["parts"], spec))
    finally:
        G.MATERIALS[BUMP] = saved
    return spec, out


def main() -> None:
    base = dict(G.MATERIALS[BUMP])
    print(f"bump layer as shipped: kxy {base['kx']:.2f}  kz {base['kz']:.2f} W/(m K)")
    for t, cu in VARIANTS:
        kxy_e, kz_e = beol_k(cu)
        print(f"BEOL {t:4.1f} um @ {cu:.0%} Cu: kxy {kxy_e:6.1f}  kz {kz_e:5.2f}")
    print()
    print(f"{'design':36s}{'rise':>8}{'worst shift':>13}{'% rise':>8}{'mesh band':>11}"
          f"{'budget':>9}{'/mesh':>7}")
    rows = []
    for case in SWEEP:
        spec, ref = peak(case, base)
        t_bump = spec.stack_um[1]
        rise = ref["metrics"]["rise_k"]
        u = ref["metrics"]["uncertainty"]
        shifts = []
        for t, cu in VARIANTS:
            _, out = peak(case, compacted(base, t_bump, t, cu))
            shifts.append({"beol_um": t, "copper_fraction": cu,
                           "shift_k": out["peak"]["t_c"] - ref["peak"]["t_c"]})
        worst = max(shifts, key=lambda s: abs(s["shift_k"]))
        rows.append({"name": case["name"], "cooling": case["cooling"], "rise_k": rise,
                     "peak_c": ref["peak"]["t_c"], "shifts": shifts,
                     "worst_shift_k": worst["shift_k"], "worst_pct_of_rise": worst["shift_k"] / rise * 100,
                     "mesh_k": u["mesh_k"], "combined_k": u["combined_k"],
                     "shift_over_mesh": abs(worst["shift_k"]) / u["mesh_k"] if u["mesh_k"] else None})
        r = rows[-1]
        print(f"{case['name'][:34]:36s}{rise:>7.2f}K{r['worst_shift_k']:>+12.3f}K"
              f"{r['worst_pct_of_rise']:>+7.2f}%{u['mesh_k']:>10.3f}K{u['combined_k']:>8.2f}K"
              f"{r['shift_over_mesh']:>7.2f}")

    worst = max(rows, key=lambda r: abs(r["worst_pct_of_rise"]))
    worst_mesh = max(rows, key=lambda r: r["shift_over_mesh"] or 0)
    report = {
        "what": "peak shift from compacting a BEOL (5-15 um, 20-40 % Cu) into the bump layer, "
                "over the fourteen designs tools/band_check.py audits",
        "treatment": "k.t along and t/k through preserved, in the bump layer's own thickness",
        "variants": [{"beol_um": t, "copper_fraction": cu} for t, cu in VARIANTS],
        "rows": rows,
        "worst_pct_of_rise": worst["worst_pct_of_rise"], "worst_design": worst["name"],
        "worst_shift_over_mesh_band": worst_mesh["shift_over_mesh"],
        "worst_over_mesh_design": worst_mesh["name"],
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nworst shift {worst['worst_pct_of_rise']:+.2f} % of the rise ({worst['name']}); "
          f"largest against the mesh band {worst_mesh['shift_over_mesh']:.2f}x ({worst_mesh['name']})")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
