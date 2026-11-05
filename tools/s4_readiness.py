"""
tools/s4_readiness.py
ThermalForge :: what the AI layer starts from, measured.

S4 is gated on S1 because a model trained on labels nobody can defend inherits
the problem, and its own gate is written in S1's units: "surrogate error at the
95th percentile sits inside the S1 error bar on held-out packages". So before
any model is written, two numbers have to exist, and this measures both on the
verified solver itself:

    the bar      the combined error band (mesh, materials, boundary -- the band
                 that tools/ttc_crossvalidate.py confirmed against measured
                 silicon) on designs spread across what the app permits. A
                 surrogate is only useful for a design if it is inside THAT
                 design's band, so the bar is a distribution, not one number.
    the labels   verified labels per second -- each one a solve whose residual,
                 energy balance and convective floor all passed, with its band
                 and its checks written beside it -- and how long a training
                 set therefore takes on this machine.

Labels are generated the way a dataset would be: many layouts per package,
because a new package costs a factorization and a new layout costs one
back-substitution.

A label is only as good as its mesh. At the standard mesh, 42 % of random
designs put a hotspot finer than the cell -- every one a patterned part on a
package of 20 mm or more, where the cell is 0.5-1.0 mm -- and the resolution
guard flags each peak as a lower bound. A surrogate trained on those learns
the mesh's error. So a flagged label is re-solved at the fine mesh, and one
still flagged there is excluded from the clean set (kept in the file, marked).
The clean-label rate is the number S4 plans on.

The split is by package, never by layout, because the gate says held-out
PACKAGES and a surrogate that has seen the package has seen most of
the answer.

    python tools/s4_readiness.py                    # 16 packages x 24 layouts
    python tools/s4_readiness.py --packages 4 --layouts 8

Writes validation/s4_readiness.json and a versioned label sample,
validation/s4_labels_v0.jsonl (inputs, peak, band and checks per label; the
fields themselves are regenerable from the inputs and are not stored).
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402

for name in ("ThermalForge.ThermalSolver", "ThermalForge.GridBuilder", "ThermalForge.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT = ROOT / "validation" / "s4_readiness.json"
LABELS = ROOT / "validation" / "s4_labels_v0.jsonl"
SCHEMA_VERSION = 0

SIZES_MM = (10.0, 20.0, 30.0, 40.0)
# cooling stack -> plausible range of the coefficient on its outer face
COOLING_H = {"direct": (500.0, 5000.0), "lid": (1000.0, 10000.0),
             "lid_heatsink": (2000.0, 30000.0), "coldplate": (10000.0, 50000.0)}
PART_SIDES_MM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
PATTERNS = tuple(E.POWER_PATTERNS)
HELD_OUT_FRACTION = 0.25


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort outside a checkout
        return "unknown"


def random_package(rng) -> dict:
    cooling = str(rng.choice(sorted(COOLING_H)))
    lo, hi = COOLING_H[cooling]
    return {"size_mm": float(rng.choice(SIZES_MM)),
            "h_w_m2k": float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), 0)),
            "ambient_c": float(np.round(rng.uniform(25.0, 45.0), 1)),
            "pitch_mm": 1.0, "cooling": cooling}


def random_parts(rng, size_mm: float, max_parts: int = 8):
    """1..max_parts distinct parts, legal (no overlap, 1 mm clear), on whole mm."""
    want = int(rng.integers(1, max_parts + 1))
    parts, taken, tries = [], [], 0
    while len(parts) < want and tries < 4000:
        tries += 1
        w, h = (float(rng.choice(PART_SIDES_MM)) for _ in range(2))
        if w > size_mm or h > size_mm:
            continue
        x = float(rng.integers(0, int(size_mm - w) + 1))
        y = float(rng.integers(0, int(size_mm - h) + 1))
        if any(x < ox + ow + 1 and ox < x + w + 1 and y < oy + oh + 1 and oy < y + h + 1
               for ox, oy, ow, oh in taken):
            continue
        taken.append((x, y, w, h))
        n = len(parts)
        parts.append({"id": f"p{n}", "name": f"P{n}", "w_mm": w, "h_mm": h,
                      "power_w": float(np.round(rng.uniform(0.5, 15.0), 2)),
                      "power_pattern": str(rng.choice(PATTERNS)), "x_mm": x, "y_mm": y})
    return parts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packages", type=int, default=16)
    ap.add_argument("--layouts", type=int, default=24)
    ap.add_argument("--mesh", default="standard", choices=sorted(E.MESH_LEVELS))
    ap.add_argument("--no-escalate", action="store_true",
                    help="keep flagged labels at the first mesh instead of re-solving them finer")
    args = ap.parse_args()

    rng = np.random.default_rng(20260922)
    cache = E.ModelCache(capacity=2)
    labels, builds, refused = [], [], {"package": 0, "layout": 0}
    commit = git_commit()
    print(f"S4 readiness: {args.packages} packages x {args.layouts} layouts, {args.mesh} mesh, "
          f"commit {commit}\n")
    print(f"{'pkg':>4}{'size':>6}  {'cooling':13s}{'h':>8}{'build':>8}{'labels':>8}"
          f"{'solve p50':>11}{'band/rise':>11}")
    t_all = time.perf_counter()
    p_index = 0
    while p_index < args.packages:
        raw = random_package(rng)
        raw["mesh"] = args.mesh
        try:
            spec = E.PackageSpec.parse(raw)
        except E.StudioError:
            refused["package"] += 1
            continue
        t0 = time.perf_counter()
        cache.get(spec)
        build_s = time.perf_counter() - t0
        builds.append(build_s)
        got, solve_ms = 0, []
        for _ in range(args.layouts):
            parts = random_parts(rng, spec.size_mm)
            try:
                chips = E.parse_chiplets(parts, spec)
            except E.StudioError:
                refused["layout"] += 1
                continue
            t0 = time.perf_counter()
            out = E.solve(cache, spec, chips)    # raises if any per-solve check fails
            mesh, first_flagged = args.mesh, bool((out.get("metrics", {}).get("uncertainty") or {})
                                                   .get("peak_is_lower_bound"))
            if out.get("ok") and first_flagged and not args.no_escalate and args.mesh != "fine":
                fine = E.PackageSpec.parse(dict(raw, mesh="fine"))
                if fine.cell_mm < spec.cell_mm:
                    out = E.solve(cache, fine, E.parse_chiplets(parts, fine))
                    mesh = "fine"
            ms = (time.perf_counter() - t0) * 1e3
            if not out.get("ok"):
                refused["layout"] += 1
                continue
            u = out["metrics"]["uncertainty"] or {}
            rise = out["metrics"]["rise_k"]
            labels.append({
                "schema": SCHEMA_VERSION, "commit": commit, "package_id": p_index,
                "package": raw, "chiplets": parts,
                "peak_c": out["peak"]["t_c"], "rise_k": rise,
                "chiplet_peaks_c": {r["id"]: r["max_c"] for r in out["chiplets"]},
                "band_k": u.get("combined_k"), "band_terms_k": {k: u.get(k) for k in
                                                               ("mesh_k", "material_k", "boundary_k")},
                "peak_is_lower_bound": bool(u.get("peak_is_lower_bound")),
                "flagged_at_first_mesh": first_flagged, "escalated": mesh != args.mesh,
                "clean": not bool(u.get("peak_is_lower_bound")),
                "checks": {"residual": out["verification"]["residual"],
                           "energy_imbalance_k": out["verification"]["energy_imbalance_k"],
                           "passed": True},
                "mesh": mesh, "unknowns": out["grid"]["unknowns"], "solve_ms": ms,
            })
            solve_ms.append(ms)
            got += 1
        rel = [lab["band_k"] / lab["rise_k"] for lab in labels[-got:] if lab["rise_k"] > 0] if got else []
        print(f"{p_index:>4}{spec.size_mm:>6.0f}  {spec.cooling:13s}{spec.h_w_m2k:>8.0f}"
              f"{build_s:>7.2f}s{got:>8}{np.median(solve_ms) if solve_ms else float('nan'):>9.1f}ms"
              f"{np.median(rel) * 100 if rel else float('nan'):>10.1f}%", flush=True)
        p_index += 1
    wall_s = time.perf_counter() - t_all

    rise = np.array([lab["rise_k"] for lab in labels])
    band = np.array([lab["band_k"] for lab in labels])
    rel = band / rise
    solve = np.array([lab["solve_ms"] for lab in labels])
    lower = sum(lab["peak_is_lower_bound"] for lab in labels)
    first = sum(lab["flagged_at_first_mesh"] for lab in labels)
    escalated = sum(lab["escalated"] for lab in labels)
    clean = len(labels) - lower
    per_label_s = wall_s / len(labels)
    per_clean_s = wall_s / clean if clean else float("inf")
    held = sorted(rng.choice(args.packages, size=max(1, int(round(HELD_OUT_FRACTION * args.packages))),
                             replace=False).tolist())

    report = {
        "what": "the bar a surrogate must beat and the rate verified labels can be made, "
                "measured on the solver before any model is written",
        "commit": commit, "mesh": args.mesh,
        "gate_s4": ("surrogate error at the 95th percentile inside the S1 error bar on held-out "
                    "packages; guided search scores at least 5x fewer layouts and returns the "
                    "identical optimum; the copilot cannot state a temperature not in a solve record"),
        "bar": {
            "definition": ("per design, the combined band (mesh, materials, boundary in quadrature) "
                           "the app prints beside the peak -- confirmed against measured silicon by "
                           "tools/ttc_crossvalidate.py (boundary term 5.5 % predicted, 5.3 % RMS "
                           "measured)"),
            "band_k": {"p5": float(np.percentile(band, 5)), "median": float(np.median(band)),
                       "p95": float(np.percentile(band, 95))},
            "band_pct_of_rise": {"p5": float(np.percentile(rel, 5) * 100),
                                 "median": float(np.median(rel) * 100),
                                 "p95": float(np.percentile(rel, 95) * 100)},
            "rise_k": {"min": float(rise.min()), "median": float(np.median(rise)),
                       "max": float(rise.max())},
            "labels_flagged_at_first_mesh": first,
            "labels_escalated_to_fine": escalated,
            "labels_flagged_lower_bound": lower,
            "note": ("a label whose peak is flagged as a lower bound (hotspot finer than the mesh) "
                     "must be excluded from training or carried with its flag"),
        },
        "labels": {
            "count": len(labels), "packages": args.packages, "layouts_per_package": args.layouts,
            "refused": refused,
            "package_build_s": {"median": float(np.median(builds)), "max": float(np.max(builds))},
            "warm_solve_ms": {"median": float(np.median(solve)), "p95": float(np.percentile(solve, 95))},
            "wall_s": wall_s, "labels_per_s": len(labels) / wall_s,
            "clean": clean, "clean_per_s": clean / wall_s,
            "hours_for_10k_clean": 1e4 * per_clean_s / 3600,
            "hours_for_100k_clean": 1e5 * per_clean_s / 3600,
            "all_checks_passed": True,
            "escalation": ("off" if args.no_escalate else
                           "a label flagged as a lower bound is re-solved at the fine mesh"),
        },
        "split": {"by": "package", "held_out_packages": held,
                  "why": "the S4 gate says held-out PACKAGES; a model that has seen the package "
                         "has seen most of the answer"},
        "label_schema": sorted(labels[0]),
        "label_file": str(LABELS.relative_to(ROOT)).replace("\\", "/"),
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with LABELS.open("w", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")

    b = report["bar"]["band_pct_of_rise"]
    lab = report["labels"]
    print(f"\n{len(labels)} verified labels in {wall_s:.1f} s -- {lab['labels_per_s']:.1f} labels/s; "
          f"package build median {lab['package_build_s']['median']:.2f} s, warm solve median "
          f"{lab['warm_solve_ms']['median']:.1f} ms")
    print(f"flagged at {args.mesh}: {first}; re-solved at fine: {escalated}; still a lower bound: "
          f"{lower}; clean: {clean} ({lab['clean_per_s']:.1f}/s)")
    print(f"100k CLEAN labels at this mix: {lab['hours_for_100k_clean']:.1f} h on this machine")
    print(f"the bar: band is {b['p5']:.1f} / {b['median']:.1f} / {b['p95']:.1f} % of the rise "
          f"(p5 / median / p95)")
    print(f"written: {OUT}\n         {LABELS}")


if __name__ == "__main__":
    main()
