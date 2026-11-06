"""
tools/electrothermal_case.py
ThermalForge :: one case, from a transistor bias to a temperature field.

Everything upstream of this tool takes watts as given. This one does not: it
states a bias instead, lets src/device.py turn the bias into watts, hands those
to the verified package solver, reads the junction back out, and goes round
again until the two agree. The number it prints is the power a design actually
draws at the temperature it actually reaches, rather than the power somebody
typed into a config.

    python tools/electrothermal_case.py                 # the demo case
    python tools/electrothermal_case.py --scenario datacenter-coldplate
    python tools/electrothermal_case.py --no-accelerate # plain fixed point
    python tools/electrothermal_case.py --sweep         # bias against runaway

What to read in the output. The pass table shows the power climbing (or
falling) as the junction moves: that is the coupling, and on a part biased
near threshold it is the thing a watts-in model cannot show you at all. The
loop gain under it is why the table ends: below 1 the fixed point is stable,
and at or above 1 there is no steady state to print, so the tool refuses.

The --sweep mode is the useful one for a design review. It raises the current
through one switch until the loop stops having a fixed point, and prints the
last bias that did -- the margin between the operating point and runaway,
measured rather than assumed.

WHAT THIS DOES NOT CLAIM. The package physics keeps its own labels unchanged.
The device model is a textbook level-1 compact model with the parameters this
file supplies, and no measured I-V curve stands behind it (src/device.py says
so at length). So the case is a verified solve driven by a modelled source,
and the tool prints that sentence rather than leaving it to be assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402
import src.electrothermal as ET  # noqa: E402
from src.device import LinearSwitch, Mosfet  # noqa: E402
from studio.scenarios import scenarios  # noqa: E402

OUT_DIR = ROOT / "validation"


def demo_sources(part_ids: List[str]) -> List:
    """A bias for the demo case, chosen so the two feedback directions both
    appear: a logic block near threshold (power rises with heat) and a
    conducting switch whose resistance drifts up (power rises too, harder).

    The MOSFET's W/L is set so the cold power lands near the scenario's own
    figure for that part, which keeps the case comparable with the
    watts-in result rather than being a different design.
    """
    out: List = []
    logic = Mosfet(vth0_v=0.4, kp_a_v2=200e-6, w_over_l=91_700, lambda_1_v=0.02)
    switch = LinearSwitch(rds0_ohm=0.030, alpha_per_k=4.0e-3)
    if len(part_ids) >= 1:
        out.append(ET.SwitchSource(part_ids[0], switch, current_a=10.0))
    if len(part_ids) >= 4:
        out.append(ET.MosfetSource(part_ids[3], logic, vgs_v=1.0, vds_v=0.8))
    elif len(part_ids) >= 2:
        out.append(ET.MosfetSource(part_ids[1], logic, vgs_v=1.0, vds_v=0.8))
    return out


def load(scenario_id: str) -> Tuple:
    by_id = {s["id"]: s for s in scenarios()}
    if scenario_id not in by_id:
        raise SystemExit(f"no scenario '{scenario_id}'; have: "
                         + ", ".join(sorted(by_id)))
    s = by_id[scenario_id]
    spec = E.PackageSpec.parse(s["package"])
    parts = E.parse_chiplets(s["chiplets"], spec)
    return s, spec, parts


def run_case(scenario_id: str, accelerate: bool) -> dict:
    s, spec, parts = load(scenario_id)
    sources = demo_sources([c.id for c in parts])
    if not sources:
        raise SystemExit(f"scenario '{scenario_id}' has no part to bias")

    cache = E.ModelCache(capacity=2)
    cold = {src.chiplet_id: src.power_w(spec.ambient_c) for src in sources}

    t0 = time.time()
    out = ET.solve_coupled(cache, spec, parts, sources, accelerate=accelerate)
    wall = time.time() - t0

    print(f"\nScenario: {s['id']} -- {s.get('title', '')}")
    print(f"  package {spec.size_mm:.0f} x {spec.size_y:.0f} mm, "
          f"h = {spec.h_w_m2k:.0f} W/(m2 K), ambient {spec.ambient_c:.0f} degC")
    print("  biased parts:")
    for src in sources:
        print(f"    {src.chiplet_id}: {src.describe()}")
        print(f"      cold power at ambient: {cold[src.chiplet_id]:.3f} W")
    print()
    for line in ET.summary_lines(out):
        print(line)

    print()
    print(f"  Power moved by the coupling, cold -> settled:")
    for src in sources:
        cid = src.chiplet_id
        hot = out["power_w"][cid]
        c = cold[cid]
        print(f"    {cid}: {c:7.3f} -> {hot:7.3f} W  "
              f"({100.0 * (hot - c) / c:+.1f} %), junction "
              f"{out['junction_c'][cid]:.2f} degC")
    peak = out["solve"]["peak"]
    print(f"  Field: peak {peak['t_c']:.2f} degC at "
          f"({peak['x_mm']:.2f}, {peak['y_mm']:.2f}) mm in the {peak['layer']}, "
          f"{out['solve']['t_min_c']:.2f} degC coldest cell")
    print(f"  Wall clock: {wall:.2f} s for {out['pass_count']} solve(s)")

    return {
        "scenario": s["id"],
        "accelerated": out["accelerated"],
        "converged": out["converged"],
        "pass_count": out["pass_count"],
        "tol_k": out["tol_k"],
        "cold_power_w": cold,
        "settled_power_w": out["power_w"],
        "junction_c": out["junction_c"],
        "loop_gain": out["loop_gain"],
        "peak_c": float(peak["t_c"]),
        "sources": out["sources"],
        "hotspots_resolved": out["hotspots_resolved"],
        "wall_s": wall,
        "label": out["label"],
    }


def run_sweep(scenario_id: str) -> dict:
    """Raise the current through one switch until the loop has no fixed point.

    Three outcomes, kept apart because they mean different things:

      settled    the loop found a steady state inside its pass budget.
      ran out    the loop is still contracting -- gain below 1, so a steady
                 state exists -- but five passes were not enough to reach the
                 tolerance. A budget problem, not a device problem.
      runaway    the gain is at or above 1, or the junction left the range the
                 device model describes. There is no steady state to report.

    The last bias that SETTLES, and the first that RUNS AWAY, are two different
    margins, and a design review wants both. A watts-in model can produce
    neither, because in a watts-in model nothing depends on temperature.
    """
    s, spec, parts = load(scenario_id)
    cid = parts[0].id
    cache = E.ModelCache(capacity=2)
    switch = LinearSwitch(rds0_ohm=0.030, alpha_per_k=4.0e-3)

    rows: List[dict] = []
    last_settled: dict = {}
    first_ran_out: dict = {}
    first_runaway: dict = {}
    current = 10.0
    while current <= 60.0:
        src = ET.SwitchSource(cid, switch, current_a=current)
        row = {"current_a": current, "cold_power_w": src.power_w(spec.ambient_c)}
        try:
            out = ET.solve_coupled(cache, spec, parts, [src])
            g = out["loop_gain"][cid]
            row.update({"outcome": "settled" if out["converged"] else "ran out",
                        "junction_c": out["junction_c"][cid],
                        "settled_power_w": out["power_w"][cid],
                        "loop_gain": g["loop_gain"],
                        "stable": g["stable"],
                        "passes_needed_estimate": out.get("passes_needed_estimate")})
            if out["converged"]:
                last_settled = row
            elif not first_ran_out:
                first_ran_out = row
        except ET.ThermalRunaway as exc:
            row.update({"outcome": "runaway", "reason": str(exc)})
            rows.append(row)
            first_runaway = row
            break
        rows.append(row)
        current += 5.0

    print(f"\nBias sweep on '{cid}' -- current against the two margins:")
    print(f"  {'I (A)':>7} {'cold W':>8} {'drawn W':>9} {'Tj (degC)':>10} "
          f"{'gain':>7}  outcome")
    for r in rows:
        if r["outcome"] == "runaway":
            print(f"  {r['current_a']:7.1f} {r['cold_power_w']:8.3f} "
                  f"{'--':>9} {'--':>10} {'--':>7}  RUNAWAY: no steady state")
            continue
        note = "settled" if r["outcome"] == "settled" else (
            f"ran out of passes (needs about {r['passes_needed_estimate']})"
            if r.get("passes_needed_estimate") else "ran out of passes")
        print(f"  {r['current_a']:7.1f} {r['cold_power_w']:8.3f} "
              f"{r['settled_power_w']:9.3f} {r['junction_c']:10.2f} "
              f"{r['loop_gain']:7.3f}  {note}")
    if last_settled:
        print(f"  settles up to {last_settled['current_a']:.1f} A "
              f"({last_settled['settled_power_w']:.2f} W drawn against "
              f"{last_settled['cold_power_w']:.2f} W cold, junction "
              f"{last_settled['junction_c']:.2f} degC, gain "
              f"{last_settled['loop_gain']:.3f})")
    if first_ran_out:
        print(f"  from {first_ran_out['current_a']:.1f} A the five-pass budget is "
              f"not enough, though the gain is still "
              f"{first_ran_out['loop_gain']:.3f} and a steady state exists")
    if first_runaway:
        print(f"  no steady state at all from {first_runaway['current_a']:.1f} A")
    else:
        print("  no runaway inside the swept range")
    return {"scenario": s["id"], "chiplet": cid, "rows": rows,
            "last_settled": last_settled or None,
            "first_ran_out": first_ran_out or None,
            "first_runaway": first_runaway or None}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One case from a transistor bias to a temperature field.")
    parser.add_argument("--scenario", default="edge-ai-soc",
                        help="scenario id (default: edge-ai-soc)")
    parser.add_argument("--no-accelerate", action="store_true",
                        help="plain fixed-point iteration, no extrapolation")
    parser.add_argument("--sweep", action="store_true",
                        help="raise the bias until the loop stops settling")
    parser.add_argument("--write", action="store_true",
                        help="write the result to validation/electrothermal_case.json")
    args = parser.parse_args()

    if args.sweep:
        report = run_sweep(args.scenario)
        name = "electrothermal_sweep.json"
    else:
        report = run_case(args.scenario, accelerate=not args.no_accelerate)
        name = "electrothermal_case.json"

    print(f"\n  {ET.solve_coupled.__doc__.splitlines()[0]}")
    if args.write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUT_DIR / name
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  written: {target}")


if __name__ == "__main__":
    main()
