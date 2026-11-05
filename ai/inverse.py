"""
ai/inverse.py
ThermalForge :: propose the package from the constraints (roadmap S4, step 4.5).

Given the parts, a temperature limit and (optionally) which parts talk to
each other, find the smallest package and the lightest cooling stack that
meet the limit, and the trade-off between peak temperature, package area and
interconnect length -- with every point on that trade-off a PROVEN optimum.

    1. screen    every candidate (package size x cooling x interposer) gets a
                 layout from a greedy search on the reduced-order model's unit
                 fields, scored by the learned surrogate. Milliseconds each,
                 so the whole grid is cheap -- and it is only a ranking.
    2. exact     the survivors -- the smallest few that could meet the limit
                 within the surrogate's own conformal interval, and the best
                 few whatever their size -- are optimized by the exact search.
    3. front     peak, area and interconnect length of those optima; the ones
                 no other candidate beats on all three are the Pareto front.
                 For the proposal, an epsilon-constraint sweep on interconnect
                 length adds exact points trading temperature for wire.

The screen can only cost optimality of the PROPOSAL (a candidate wrongly
screened out is never run), never correctness: every number reported is from
an exact, re-solved optimization. The screen's accuracy is measured on the
candidates that were run, and reported beside the result.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ai import records as rec
from ai import rom as R

SIZES_MM = (10.0, 15.0, 20.0, 30.0, 40.0)
COOLINGS = (("direct", 2500.0), ("lid", 2500.0), ("lid_heatsink", 5000.0), ("coldplate", 25000.0))
INTERPOSERS_UM = (100.0, 200.0)
COOLING_COST = {"direct": 0, "lid": 1, "lid_heatsink": 2, "coldplate": 3}


def pitch_for(size_mm: float) -> float:
    return 1.0 if size_mm <= 15 else (2.0 if size_mm <= 30 else 4.0)


def _greedy_rom(stack: "R.PlateStack", spec, parts: List[dict], gap_mm: Optional[float] = None):
    """A good layout on the ROM, by superposition: every part at every pitch
    position, placed hottest first where it adds least to the peak, then one
    move at a time while that helps. Returns positions (x_mm, y_mm) or None."""
    import studio.engine as E
    pitch = spec.pitch_mm
    if gap_mm is None:
        gap_mm = spec.keepout_enforced_um / 1000.0     # the package's own keep-out, as enforced
    fields, cands = [], []
    for p in parts:
        xs = np.arange(0.0, spec.size_mm - p["w_mm"] + 1e-9, pitch)
        ys = np.arange(0.0, spec.size_y - p["h_mm"] + 1e-9, pitch)
        pos = [(float(x), float(y)) for x in xs for y in ys]
        if not pos:
            return None
        rows = []
        for x, y in pos:
            c = E.parse_chiplets([dict(p, x_mm=x, y_mm=y)], spec)[0]
            rows.append(stack.device_rise(R.part_power_map(c, stack.nx, stack.ny, stack.cell_mm)).ravel())
        fields.append(np.asarray(rows))
        cands.append(pos)

    def clash(a, pa, b, pb):
        (xa, ya), (xb, yb) = pa, pb
        return (xa < xb + parts[b]["w_mm"] + gap_mm - 1e-9 and xb < xa + parts[a]["w_mm"] + gap_mm - 1e-9
                and ya < yb + parts[b]["h_mm"] + gap_mm - 1e-9 and yb < ya + parts[a]["h_mm"] + gap_mm - 1e-9)

    order = sorted(range(len(parts)), key=lambda i: -parts[i]["power_w"])
    choice: Dict[int, int] = {}
    running = np.zeros(fields[0].shape[1])
    for i in order:
        ok = [j for j, pj in enumerate(cands[i])
              if all(not clash(i, pj, o, cands[o][choice[o]]) for o in choice)]
        if not ok:
            return None
        j = ok[int(np.argmin((fields[i][ok] + running).max(axis=1)))]
        choice[i] = j
        running += fields[i][j]
    value = float(running.max())
    for _ in range(20):
        improved = False
        for i in range(len(parts)):
            others = running - fields[i][choice[i]]
            ok = [j for j, pj in enumerate(cands[i])
                  if all(not clash(i, pj, o, cands[o][choice[o]]) for o in choice if o != i)]
            peaks = (fields[i][ok] + others).max(axis=1)
            j = ok[int(np.argmin(peaks))]
            if peaks.min() < value - 1e-9:
                choice[i], running, value = j, others + fields[i][j], float(peaks.min())
                improved = True
        if not improved:
            break
    return [cands[i][choice[i]] for i in range(len(parts))]


@dataclass
class Candidate:
    size_mm: float
    cooling: str
    h_w_m2k: float
    interposer_um: float
    package: dict
    screen_peak_c: Optional[float] = None
    screen_half_k: Optional[float] = None
    screen_layout: Optional[List[Tuple[float, float]]] = None
    exact: Optional[rec.SolveRecord] = None

    @property
    def area_mm2(self) -> float:
        return self.size_mm * self.size_mm

    def label(self) -> str:
        return (f"{self.size_mm:g} mm, {self.cooling.replace('_', ' + ')} (h {self.h_w_m2k:g}), "
                f"{self.interposer_um:g} um interposer")


def candidates(parts: List[dict], base_package: dict) -> List[Candidate]:
    area = sum(p["w_mm"] * p["h_mm"] for p in parts)
    biggest = max(max(p["w_mm"], p["h_mm"]) for p in parts)
    out = []
    for size, (cooling, h), t_i in itertools.product(SIZES_MM, COOLINGS, INTERPOSERS_UM):
        if size < biggest + 1 or area > 0.55 * size * size:
            continue                       # cannot hold the parts with room to move
        stack = list(base_package.get("stack_um", [100.0, 40.0, 100.0]))
        stack[0] = t_i
        pkg = {"size_mm": size, "h_w_m2k": h, "ambient_c": base_package.get("ambient_c", 25.0),
               "pitch_mm": pitch_for(size), "cooling": cooling, "stack_um": stack}
        out.append(Candidate(size, cooling, h, t_i, pkg))
    return out


def screen(cands: List[Candidate], parts: List[dict], surrogate=None) -> float:
    """Screen every candidate on the ROM (+ surrogate). Returns seconds spent."""
    import studio.engine as E
    t0 = time.perf_counter()
    cache = R.RomCache(capacity=4, factor=2.0 if surrogate is None else surrogate.rom_factor)
    for c in cands:
        try:
            spec = E.PackageSpec.parse(c.package)
        except E.StudioError:
            continue
        try:
            stack = cache.get(spec)
            layout = _greedy_rom(stack, spec, parts)
            if layout is None:
                continue
            placed = [dict(p, x_mm=x, y_mm=y) for p, (x, y) in zip(parts, layout)]
            chips = E.parse_chiplets(placed, spec)
        except E.StudioError:
            continue              # e.g. the parts are not on this package's mesh grid
        if surrogate is not None:
            pred = surrogate.predict(cache, spec, chips)
            c.screen_peak_c = pred["peak_c"]
            c.screen_half_k = (pred["interval_c"][1] - pred["peak_c"]) if "interval_c" in pred else 0.0
        else:
            sol = R.solve_layout(cache, spec, chips)
            c.screen_peak_c = spec.ambient_c + max(sol.part_rise)
            c.screen_half_k = 0.15 * max(sol.part_rise)      # no calibrated interval without a model
        c.screen_layout = layout
    return time.perf_counter() - t0


def select(cands: List[Candidate], max_peak_c: Optional[float], n_small: int = 3,
           n_best: int = 2) -> List[Candidate]:
    screened = [c for c in cands if c.screen_peak_c is not None]
    if max_peak_c is not None:
        could = [c for c in screened if c.screen_peak_c - c.screen_half_k <= max_peak_c]
    else:
        could = list(screened)
    by_cost = sorted(could, key=lambda c: (c.area_mm2, COOLING_COST[c.cooling], c.interposer_um,
                                           c.screen_peak_c))
    chosen = by_cost[:n_small]
    for c in sorted(screened, key=lambda c: c.screen_peak_c)[:n_best]:
        if c not in chosen:
            chosen.append(c)
    return chosen


def pareto(points: List[Tuple[float, ...]]) -> List[bool]:
    keep = []
    for i, p in enumerate(points):
        dominated = any(all(q[k] <= p[k] for k in range(len(p))) and any(q[k] < p[k] for k in range(len(p)))
                        for j, q in enumerate(points) if j != i)
        keep.append(not dominated)
    return keep


def design(parts: List[dict], base_package: dict, store: rec.RecordStore, cache,
           max_peak_c: Optional[float] = None, links: Optional[Sequence[Tuple[str, str, float]]] = None,
           surrogate=None, n_small: int = 3, n_best: int = 2, sweep: Sequence[float] = (0.9, 0.8)) -> Dict:
    t_all = time.perf_counter()
    cands = candidates(parts, base_package)
    screen_s = screen(cands, parts, surrogate)
    chosen = select(cands, max_peak_c, n_small, n_best)
    t0 = time.perf_counter()
    for c in chosen:
        placed = [dict(p, x_mm=x, y_mm=y) for p, (x, y) in zip(parts, c.screen_layout)]
        try:
            c.exact = rec.record_optimize(store, cache, c.package, placed, search="guided",
                                          links=list(links) if links else None)
        except Exception:  # noqa: BLE001 -- a candidate the exact search refuses is not proposed
            c.exact = None
    exact_s = time.perf_counter() - t0
    done = [c for c in chosen if c.exact is not None]
    rows = []
    for c in done:
        peak = c.exact.value("optimized.peak_c")
        rows.append({"candidate": c.label(), "package": c.package, "record": c.exact.id,
                     "area_mm2": c.area_mm2, "cooling": c.cooling, "peak_c": peak,
                     "screen_peak_c": c.screen_peak_c, "screen_error_k": c.screen_peak_c - peak,
                     "meets": None if max_peak_c is None else peak <= max_peak_c})
    # interconnect length of each optimum, from its placement
    if links:
        for r, c in zip(rows, done):
            pos = {p["id"]: p for p in c.exact.value("placement")}
            r["wirelength_mm"] = float(sum(w * (abs((pos[a]["x_mm"] + pos[a]["w_mm"] / 2)
                                                    - (pos[b]["x_mm"] + pos[b]["w_mm"] / 2))
                                                + abs((pos[a]["y_mm"] + pos[a]["h_mm"] / 2)
                                                      - (pos[b]["y_mm"] + pos[b]["h_mm"] / 2)))
                                           for a, b, w in links))
    objectives = [(r["peak_c"], r["area_mm2"], COOLING_COST[r["cooling"]]) + (
        (r["wirelength_mm"],) if links else ()) for r in rows]
    for r, front in zip(rows, pareto(objectives)):
        r["pareto"] = front
    meeting = [r for r in rows if r["meets"]] if max_peak_c is not None else rows
    proposal = min(meeting, key=lambda r: (r["area_mm2"], COOLING_COST[r["cooling"]], r["peak_c"])) \
        if meeting else None
    sweep_rows = []
    if proposal is not None and links:
        base_wl = proposal["wirelength_mm"]
        prop_c = next(c for c in done if c.exact.id == proposal["record"])
        placed = [dict(p, x_mm=x, y_mm=y) for p, (x, y) in zip(parts, prop_c.screen_layout)]
        for frac in sweep:
            cap = base_wl * frac
            try:
                r2 = rec.record_optimize(store, cache, prop_c.package, placed, search="guided",
                                         links=list(links), wirelength_cap_mm=cap)
                sweep_rows.append({"cap_mm": cap, "record": r2.id,
                                   "peak_c": r2.value("optimized.peak_c"),
                                   "wirelength_mm": r2.value("certificate.wirelength_mm")})
            except Exception as exc:  # noqa: BLE001
                sweep_rows.append({"cap_mm": cap, "infeasible": str(exc)})
    errs = [abs(r["screen_error_k"]) for r in rows]
    return {"candidates": len(cands), "screened": sum(c.screen_peak_c is not None for c in cands),
            "exact_runs": len(done), "rows": rows, "proposal": proposal, "sweep": sweep_rows,
            "screen_s": screen_s, "exact_s": exact_s, "total_s": time.perf_counter() - t_all,
            "screen_abs_error_k": {"max": max(errs) if errs else None,
                                   "mean": float(np.mean(errs)) if errs else None},
            "screen_model": "surrogate" if surrogate is not None else "ROM only"}
