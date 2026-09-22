"""
tools/s3_gate.py
RISC3D-AI :: the S3 gate, checked against many layouts rather than one.

The original gate -- 30 chiplets, 40 x 40 mm, 0.25 mm, proven optimal in under
a minute -- was measured out of reach: the exact search crosses a minute at 14
chiplets (tools/search_scaling.py). The decision taken on 22 Sep 2026 was to
keep "proven optimal" rather than trade it for a multi-resolution tolerance,
and to set the gate at what the exact search reliably does.

"Reliably" is the point of this file. Search time depends on the layout, not
just on the count -- 10 parts once took 11.9 s where 12 took 8.7 s, because the
bound bites harder when parts crowd each other -- so one lucky instance proves
nothing. The gate passes only if the WORST of several random layouts does.

And the layouts have to be honest ones. tools/search_scaling.py measured 12
parts in under a minute, but its parts were identical 4 x 4 mm squares with
seven distinct powers: parts of the same size and power are interchangeable,
the search collapses them by symmetry, and 12 such parts hide five pairs. Real
floorplans rarely repeat a part exactly, so every part here gets its own size
and power, the report counts any symmetry that survives, and the gate is set
on that harder case. A trial that runs past twice the budget is cancelled and
recorded as a failure with its time, rather than left to run for hours.

    python tools/s3_gate.py --find          # largest count whose worst layout passes
    python tools/s3_gate.py --parts 8       # the gate at one count

Writes validation/s3_gate.json (standard mesh) or validation/s3_gate_<mesh>.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

BUDGET_S = 60.0
CAP_S = 2 * BUDGET_S          # a trial past this is cancelled and fails
DRAG_BUDGET_MS = 50.0


def random_layout(rng, n_parts: int, size_mm: float, pitch_mm: float):
    """n non-overlapping parts of mixed size and power, on the pitch, with the
    keep-out respected, so every instance is a legal starting layout."""
    parts, taken = [], []
    sizes = [(4.0, 4.0), (4.0, 2.0), (2.0, 2.0), (6.0, 4.0)]
    tries = 0
    while len(parts) < n_parts:
        tries += 1
        if tries > 20000:
            raise RuntimeError("could not pack the requested parts; use fewer")
        w, h = sizes[int(rng.integers(len(sizes)))]
        steps_x = int((size_mm - w) // pitch_mm) + 1
        steps_y = int((size_mm - h) // pitch_mm) + 1
        x = float(rng.integers(steps_x)) * pitch_mm
        y = float(rng.integers(steps_y)) * pitch_mm
        gap = 1.0                       # comfortably clear of the keep-out
        if any(x < ox + ow + gap and ox < x + w + gap and y < oy + oh + gap and oy < y + h + gap
               for ox, oy, ow, oh in taken):
            continue
        taken.append((x, y, w, h))
        n = len(parts)
        parts.append({"id": f"c{n}", "name": f"C{n}", "w_mm": w, "h_mm": h,
                      "power_w": float(np.round(rng.uniform(0.5, 4.0), 2)),
                      "x_mm": x, "y_mm": y})
    return parts


def drag_ms(cache, pkg, parts, rng, moves: int = 30) -> float:
    """p95 of warm re-solves after single-part moves -- what a drag costs."""
    times = []
    for _ in range(moves):
        moved = [dict(p) for p in parts]
        k = int(rng.integers(len(moved)))
        moved[k]["x_mm"] = float(max(0.0, min(pkg.size_mm - moved[k]["w_mm"],
                                              moved[k]["x_mm"] + rng.choice([-2.0, 2.0]))))
        chips = E.parse_chiplets(moved, pkg)
        t0 = time.perf_counter()
        E.solve(cache, pkg, chips)
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.percentile(times, 95))


def trial(cache, pkg, parts, rng) -> dict:
    """One random layout: the proof, capped, then what a drag costs."""
    chips = E.parse_chiplets(parts, pkg)
    shared = len(parts) - len({(p["w_mm"], p["h_mm"], p["power_w"]) for p in parts})
    cancel = threading.Event()
    timer = threading.Timer(CAP_S, cancel.set)
    timer.start()
    t0 = time.perf_counter()
    try:
        cert = E.optimize(cache, pkg, chips, cancel=cancel)["certificate"]
    except E.SearchCancelled:
        cert = None
    finally:
        timer.cancel()
    proof_s = time.perf_counter() - t0
    drag = drag_ms(cache, pkg, parts, rng)
    ok = (cert is not None and proof_s < BUDGET_S and drag < DRAG_BUDGET_MS
          and cert["resolve_delta_k"] <= E.RESOLVE_TOL_K)
    return {"parts": len(parts), "interchangeable_parts": shared,
            "combinations": None if cert is None else cert["combinations"],
            "proof_s": proof_s, "cancelled": cert is None,
            "drag_p95_ms": drag,
            "resolve_delta_k": None if cert is None else cert["resolve_delta_k"],
            "optimum_c": None if cert is None else cert["lattice_optimum_c"], "passed": ok}


def run_count(cache, pkg, args, n_parts, rng, stop_on_fail: bool):
    rows = []
    for t in range(args.trials):
        parts = random_layout(rng, n_parts, args.size, args.pitch)
        r = trial(cache, pkg, parts, rng)
        r["trial"] = t
        rows.append(r)
        combos = "cancelled" if r["cancelled"] else f"{r['combinations']:,}"
        print(f"{n_parts:>6}{t:>6}{combos:>24}{r['proof_s']:>8.1f}s{r['drag_p95_ms']:>9.1f}ms"
              f"{r['interchangeable_parts']:>6}   {'pass' if r['passed'] else 'FAIL'}", flush=True)
        if stop_on_fail and not r["passed"]:
            break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--find", action="store_true",
                    help="walk the count up from 4 and report the largest whose worst layout passes")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--size", type=float, default=40.0)
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--mesh", default="standard", choices=sorted(E.MESH_LEVELS))
    args = ap.parse_args()

    pkg = E.PackageSpec.parse({"size_mm": args.size, "h_w_m2k": 5000.0, "ambient_c": 25.0,
                               "pitch_mm": args.pitch, "mesh": args.mesh})
    cache = E.ModelCache()
    t0 = time.perf_counter()
    cache.get(pkg)
    build_s = time.perf_counter() - t0
    print(f"S3 gate: {args.size:.0f} x {args.size:.0f} mm, {pkg.cell_mm} mm cells "
          f"({pkg.unknowns:,} unknowns), pitch {args.pitch} mm, {args.trials} random layouts "
          f"per count, every part distinct; trials capped at {CAP_S:.0f} s")
    print(f"model built once in {build_s:.1f} s (cached; not part of a proof)\n")
    print(f"{'parts':>6}{'trial':>6}{'layouts':>24}{'proof':>9}{'drag p95':>11}{'twins':>6}   verdict",
          flush=True)

    rng = np.random.default_rng(20260922)
    counts = list(range(4, E.MAX_CHIPLETS + 1, 2)) if args.find else [args.parts]
    by_count, largest = {}, None
    for n in counts:
        rows = run_count(cache, pkg, args, n, rng, stop_on_fail=args.find)
        by_count[n] = rows
        if all(r["passed"] for r in rows) and len(rows) == args.trials:
            largest = n
        else:
            break

    rows = [r for n in by_count for r in by_count[n]]
    gate_n = largest if args.find else args.parts
    gate_rows = by_count.get(gate_n, [])
    passed = bool(gate_rows) and all(r["passed"] for r in gate_rows) and len(gate_rows) == args.trials
    worst_proof = max((r["proof_s"] for r in gate_rows), default=None)
    worst_drag = max((r["drag_p95_ms"] for r in gate_rows), default=None)
    report = {
        "gate": (f"{gate_n} distinct chiplets on {args.size:.0f} x {args.size:.0f} mm proven optimal "
                 f"in under {BUDGET_S:.0f} s, dragging under {DRAG_BUDGET_MS:.0f} ms, judged by the "
                 f"WORST of {args.trials} random layouts"),
        "mode": "find" if args.find else "fixed",
        "mesh": args.mesh, "cell_mm": pkg.cell_mm, "unknowns": pkg.unknowns,
        "pitch_mm": args.pitch, "trial_cap_s": CAP_S, "rows": rows,
        "largest_passing_count": largest,
        "worst_proof_s": worst_proof, "worst_drag_p95_ms": worst_drag,
        "passed": passed,
    }
    out = ROOT / "validation" / ("s3_gate.json" if args.mesh == "standard" else f"s3_gate_{args.mesh}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.find:
        print(f"\nlargest count whose worst layout passes: {largest}")
    if worst_proof is not None:
        print(f"at {gate_n}: worst proof {worst_proof:.1f} s, worst drag p95 {worst_drag:.1f} ms  ->  "
              f"{'GATE PASSED' if passed else 'GATE FAILED'}")
    print(f"written: {out}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
