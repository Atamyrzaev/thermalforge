"""
tools/band_check.py
ThermalForge :: is the published mesh band actually a bound?

The band printed beside every peak is measured on reference packages and then
scaled to the design in hand by its rise above ambient. Scaling is an
assumption, and this is the tool that tests it: it takes a spread of designs,
solves each one at BOTH mesh levels, and compares the difference it actually
sees against the band the app quoted.

What comes out is the ratio measured/estimated. Above 1.0 means the app quoted
a band its own design exceeds, which is the failure worth knowing about --
a band that flatters is worse than no band, because somebody spends it.

    python tools/band_check.py              # the standard sweep
    python tools/band_check.py --quick      # a third of it

The sweep deliberately includes designs MORE concentrated than the reference
packages: small parts, high density, hard cooling, internal patterns. Those are
where scaling by rise is weakest, so they are where the margin has to be set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402

OUT_DIR = ROOT / "validation"


def design(name: str, size_mm: float, h: float, parts, cooling: str = "direct",
           pitch_mm: float = 1.0) -> Dict:
    return {"name": name, "size_mm": size_mm, "h": h, "parts": parts,
            "cooling": cooling, "pitch_mm": pitch_mm}


def part(cid, w, h, power, x, y, pattern="uniform"):
    return {"id": cid, "name": cid.upper(), "w_mm": w, "h_mm": h, "power_w": power,
            "x_mm": x, "y_mm": y, "power_pattern": pattern}


# Chosen to stress the scaling, not to flatter it: the reference packages are
# 20 mm with four large parts, so these go smaller, denser, harder-cooled and
# patterned, in every combination the app can build.
SWEEP: List[Dict] = [
    design("two 4 mm parts, film", 20.0, 2000.0,
           [part("a", 4, 4, 6.0, 2, 2), part("b", 4, 4, 4.0, 12, 12)], pitch_mm=2.0),
    design("two 4 mm parts, cold plate", 20.0, 25000.0,
           [part("a", 4, 4, 6.0, 2, 2), part("b", 4, 4, 4.0, 12, 12)], pitch_mm=2.0),
    design("one 2 mm part, dense", 20.0, 2000.0, [part("hot", 2, 2, 8.0, 9, 9)]),
    design("one 2 mm part, cold plate", 20.0, 25000.0, [part("hot", 2, 2, 8.0, 9, 9)]),
    design("one 2 mm part, weak film", 20.0, 500.0, [part("hot", 2, 2, 8.0, 9, 9)]),
    design("MAC array alone", 20.0, 2000.0,
           [part("npu", 5, 5, 8.0, 7, 7, "mac_array")]),
    design("MAC array, cold plate", 20.0, 25000.0,
           [part("npu", 5, 5, 8.0, 7, 7, "mac_array")]),
    design("MAC array under a lid", 20.0, 1000.0,
           [part("npu", 5, 5, 8.0, 7, 7, "mac_array")], cooling="lid_heatsink"),
    design("two engines, dense", 20.0, 4000.0,
           [part("gpu", 4, 4, 12.0, 8, 8, "two_engines")]),
    design("quad cores plus a hot edge part", 20.0, 2000.0,
           [part("cpu", 6, 6, 9.0, 1, 1, "quad_cores"), part("io", 2, 2, 4.0, 17, 17)]),
    design("crowded, mixed patterns", 20.0, 2500.0,
           [part("a", 3, 3, 4.0, 1, 1, "mac_array"), part("b", 3, 3, 4.0, 8, 1),
            part("c", 3, 3, 4.0, 1, 8, "two_engines"), part("d", 3, 3, 4.0, 8, 8)]),
    design("40 mm package, small dense part", 40.0, 2000.0,
           [part("hot", 4, 4, 10.0, 18, 18)], pitch_mm=2.0),
    design("40 mm package, patterned", 40.0, 2000.0,
           [part("npu", 8, 8, 12.0, 16, 16, "mac_array")], pitch_mm=2.0),
    design("30 mm package, cold plate", 30.0, 25000.0,
           [part("a", 5, 5, 10.0, 12, 12, "mac_array")], pitch_mm=1.0),
]


def run(quick: bool) -> Dict:
    cache = E.ModelCache()
    sweep = SWEEP[::3] if quick else SWEEP
    rows: List[Dict] = []
    for case in sweep:
        raw_pkg = {"size_mm": case["size_mm"], "h_w_m2k": case["h"], "ambient_c": 25.0,
                   "pitch_mm": case["pitch_mm"], "cooling": case["cooling"]}
        spec = E.PackageSpec.parse(raw_pkg)
        chiplets = E.parse_chiplets(case["parts"], spec)
        started = time.perf_counter()
        out = E.mesh_check(cache, spec, chiplets)
        if out["same_mesh"]:
            print(f"  {case['name'][:34]:36s} already at the finest cell; skipped")
            continue
        coarse, fine = out["levels"]
        ratio = (out["measured_k"] / out["estimated_k"]) if out["estimated_k"] else None
        rows.append({
            "name": case["name"],
            "size_mm": case["size_mm"], "h_w_m2k": case["h"], "cooling": case["cooling"],
            "concentrated": E.any_internal_power(chiplets),
            "cell_coarse_mm": coarse["cell_mm"], "cell_fine_mm": fine["cell_mm"],
            "peak_coarse_c": coarse["peak_c"], "peak_fine_c": fine["peak_c"],
            "measured_k": out["measured_k"], "estimated_k": out["estimated_k"],
            "ratio": ratio, "seconds": time.perf_counter() - started,
        })
        flag = "  <-- EXCEEDS THE BAND" if ratio and ratio > 1.0 else ""
        print(f"  {case['name'][:34]:36s} measured {out['measured_k']:7.3f} K   "
              f"quoted {out['estimated_k']:7.3f} K   ratio {ratio:5.2f}{flag}")
    worst = max((r for r in rows if r["ratio"] is not None),
                key=lambda r: r["ratio"], default=None)
    return {
        "what": "measured mesh error against the band the app quoted, over a sweep "
                "of designs chosen to stress the scaling rather than flatter it",
        "designs": len(rows),
        "rows": rows,
        "worst_ratio": worst["ratio"] if worst else None,
        "worst_design": worst["name"] if worst else None,
        "exceeded": [r["name"] for r in rows if r["ratio"] and r["ratio"] > 1.0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="a third of the sweep")
    args = parser.parse_args()

    print("Measured mesh error against the quoted band:")
    report = run(args.quick)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "band_check.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}")
    if report["worst_ratio"] is None:
        print("  nothing measured")
        return
    print(f"  worst ratio {report['worst_ratio']:.2f} on '{report['worst_design']}'")
    if report["exceeded"]:
        print(f"  EXCEEDED on {len(report['exceeded'])} design(s): "
              + ", ".join(report["exceeded"]))
        print("  the transfer margin in studio/engine.py must cover this")
    else:
        print("  every design came in under the band it was quoted")


if __name__ == "__main__":
    main()
