"""
electrothermal.py
ThermalForge :: the loop between what the silicon draws and how hot it gets.

The package solver answers one question: given these watts, what temperature?
src/device.py answers the other: given this temperature, how many watts? Put
together they are a fixed point, and this file finds it -- or says that it
could not, which for a positive loop is the honest answer rather than a
failure to work around.

    power at an assumed junction  ->  solve the package  ->  junction
            ^                                                   |
            +---------------------------------------------------+

The direction of the loop is the whole point. A device biased near threshold
draws MORE current when it is hot (the threshold falls faster than the
mobility), so the loop amplifies: each pass raises the power, which raises the
junction, which raises the power. A device biased hard draws LESS, and the loop
damps. Which of the two a design is in is not a matter of opinion, so this file
measures it: the loop gain at the converged point, as the product of how much
power a kelvin buys (dP/dT, from the device) and how many kelvin a watt buys
(the thermal resistance, from the solve). Below 1 the fixed point is stable.
At or above 1 there is no steady state to report, and it raises.

WHAT THIS LAYER CLAIMS. Nothing about silicon. The package solve inside the
loop carries its own verification and its own validation label, unchanged --
this file does not touch the physics, it only chooses the watts handed to it.
The device model is a textbook compact model with the caller's parameters and
no measured I-V behind it (see src/device.py). So a coupled answer is
"verified, with a power source that is modelled rather than measured", and the
label it carries says exactly that.

One approximation is worth naming because it is not obvious. The junction
temperature fed back to a device is that chiplet's hottest resolved cell, which
is a cell average over 20 um of silicon, not a device-layer temperature. For a
part whose power is spread over the footprint the two are close; for a part
whose hotspot is finer than the mesh, the solve already fails its own
"hotspots resolved" check and labels its peak a lower bound, and a lower bound
fed into a positive loop under-states the power. The loop reports the solve's
resolution verdict for exactly this reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.device import MAX_JUNCTION_C, DeviceError, LinearSwitch, Mosfet

#: Junction change between passes below which the fixed point is called found.
#: 0.05 K is well inside every band the package solver quotes, so a tighter
#: tolerance would be converging on noise the thermal model cannot resolve.
DEFAULT_TOL_K = 0.05

#: The plan's budget: a simplified feedback is allowed five passes, and after
#: that the answer is "did not settle" rather than a sixth pass.
DEFAULT_MAX_PASSES = 5


class CouplingError(Exception):
    """The loop cannot return a steady state it can defend."""


class ThermalRunaway(CouplingError):
    """The loop amplifies: there is no fixed point to report."""


@dataclass(frozen=True)
class MosfetSource:
    """A chiplet whose power is `count` identical MOSFETs at one bias."""

    chiplet_id: str
    device: Mosfet
    vgs_v: float
    vds_v: float
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise DeviceError(f"{self.chiplet_id}: count must be >= 1, got {self.count}")

    def power_w(self, t_c: float) -> float:
        return self.count * self.device.power_w(self.vgs_v, self.vds_v, t_c)

    def describe(self) -> str:
        return (f"{self.count} x level-1 MOSFET at Vgs={self.vgs_v:.3g} V, "
                f"Vds={self.vds_v:.3g} V")


@dataclass(frozen=True)
class SwitchSource:
    """A chiplet whose power is `count` identical switches carrying a current."""

    chiplet_id: str
    switch: LinearSwitch
    current_a: float
    count: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise DeviceError(f"{self.chiplet_id}: count must be >= 1, got {self.count}")

    def power_w(self, t_c: float) -> float:
        return self.count * self.switch.power_w(self.current_a, t_c)

    def describe(self) -> str:
        return (f"{self.count} x linear switch at {self.current_a:.3g} A, "
                f"Rds0={self.switch.rds0_ohm:.4g} ohm, "
                f"alpha={self.switch.alpha_per_k * 100:.3g} %/K")


Source = object  # MosfetSource | SwitchSource; either exposes power_w(t_c)


def _powers_at(sources: Sequence, junction_c: Dict[str, float],
               fallback_c: float) -> Dict[str, float]:
    """What each sourced chiplet draws at the junction temperature assumed for
    it. A chiplet with no temperature yet (the first pass) uses `fallback_c`."""
    out: Dict[str, float] = {}
    for s in sources:
        t = junction_c.get(s.chiplet_id, fallback_c)
        out[s.chiplet_id] = float(s.power_w(t))
    return out


def _with_powers(chiplets: Sequence, powers_w: Dict[str, float]) -> Tuple:
    """The same chiplets, with the sourced ones carrying their computed power.
    Geometry, placement and internal power pattern are untouched: only the
    total watts of a part change, which is the one thing this layer decides."""
    out = []
    for c in chiplets:
        if c.id in powers_w:
            out.append(replace(c, power_w=powers_w[c.id]))
        else:
            out.append(c)
    return tuple(out)


def _dp_dt(source, t_c: float, step_k: float = 0.5) -> float:
    """How many watts a kelvin buys at this operating point, by central
    difference on the device model. Watts per kelvin."""
    lo, hi = t_c - step_k, t_c + step_k
    if hi > MAX_JUNCTION_C:
        hi, lo = t_c, t_c - 2.0 * step_k
    return (source.power_w(hi) - source.power_w(lo)) / (hi - lo)


def _extrapolated(got: Dict[str, float], signed: Dict[str, float],
                  previous: Optional[Dict[str, float]]) -> Dict[str, float]:
    """The next junction guess, accelerated where the iteration is visibly
    geometric.

    A stable loop converges linearly: each pass closes a fixed fraction r of
    what is left, and r is the loop gain. Two successive moves measure it,
    r = d_n / d_(n-1), and the limit of the geometric series is
    T_n + d_n * r / (1 - r). That is Aitken's delta-squared, applied per
    chiplet.

    A negative r is the same series with alternating signs, which is what
    negative feedback produces -- a part biased above its zero-temperature-
    coefficient point overshoots cool, then warm, then cool. Aitken holds for
    any |r| < 1, so both are extrapolated.

    This only chooses where to evaluate next. Every temperature this module
    reports comes out of an actual solve at an actual power, so acceleration
    can cost a pass or save one but cannot invent a number. Guards: only a
    contraction is extrapolated (0 < |r| < 0.95), and never past 20 K in one
    step, because a ratio measured on two noisy moves can be wrong.
    """
    if previous is None:
        return dict(got)
    out = dict(got)
    for i, d_now in signed.items():
        d_prev = previous.get(i)
        if d_prev is None or d_prev == 0.0 or d_now == 0.0:
            continue
        r = d_now / d_prev
        if not (0.0 < abs(r) < 0.95):
            continue
        step = d_now * r / (1.0 - r)
        if abs(step) > 20.0:
            step = math.copysign(20.0, step)
        out[i] = got[i] + step
    return out


def solve_coupled(cache, spec, chiplets, sources: Sequence,
                  max_passes: int = DEFAULT_MAX_PASSES,
                  tol_k: float = DEFAULT_TOL_K,
                  start_c: Optional[float] = None,
                  accelerate: bool = True) -> dict:
    """Find the electro-thermal fixed point, or report that there is none.

    `cache`, `spec` and `chiplets` are what studio.engine.solve takes; the
    chiplets' `power_w` is overwritten for every part that has a source and
    left alone for every part that does not.

    Returns a dict carrying the final solve payload under "solve", the pass by
    pass history under "passes", whether it settled under "converged", and the
    measured loop gain per sourced chiplet under "loop_gain". Every reported
    temperature is from a solve; `accelerate` changes only which power the
    next solve is run at (see _extrapolated).

    Raises ThermalRunaway when the loop amplifies rather than settling, and
    CouplingError when the inputs cannot describe a loop at all.
    """
    import studio.engine as E

    if not sources:
        raise CouplingError("no device sources: there is no loop to solve")
    ids = {c.id for c in chiplets}
    for s in sources:
        if s.chiplet_id not in ids:
            raise CouplingError(f"source names chiplet '{s.chiplet_id}', which is not "
                                "in this placement")
    seen = [s.chiplet_id for s in sources]
    if len(set(seen)) != len(seen):
        raise CouplingError("two sources name the same chiplet; one chiplet is one "
                            "power, so the loop would not be defined")
    if max_passes < 1:
        raise CouplingError(f"max_passes must be >= 1, got {max_passes}")
    if not (tol_k > 0.0):
        raise CouplingError(f"tol_k must be > 0, got {tol_k}")

    ambient_c = float(spec.ambient_c)
    first_guess = ambient_c if start_c is None else float(start_c)

    guess: Dict[str, float] = {}
    solved_junction: Dict[str, float] = {}
    passes: List[dict] = []
    payload: dict = {}
    converged = False
    previous_signed: Optional[Dict[str, float]] = None
    previous_delta: Optional[float] = None
    growth_streak = 0

    for n in range(1, max_passes + 1):
        powers_w = _powers_at(sources, guess, first_guess)
        assumed = {s.chiplet_id: guess.get(s.chiplet_id, first_guess)
                   for s in sources}
        payload = E.solve(cache, spec, _with_powers(chiplets, powers_w))
        if not payload.get("ok"):
            raise CouplingError(
                "the placement handed to the loop cannot be solved: "
                + "; ".join(str(v) for v in payload.get("violations", [])))

        got = {row["id"]: float(row["max_c"]) for row in payload["chiplets"]}
        signed = {i: got[i] - assumed[i] for i in assumed}
        delta = max((abs(v) for v in signed.values()), default=0.0)
        passes.append({
            "pass": n,
            "assumed_junction_c": dict(assumed),
            "power_w": dict(powers_w),
            "junction_c": {i: got[i] for i in assumed},
            "peak_c": float(payload["peak"]["t_c"]),
            "total_power_w": float(sum(powers_w.values())),
            "delta_k": float(delta),
        })
        solved_junction = {i: got[i] for i in assumed}

        hottest = max(got.values()) if got else ambient_c
        if hottest > MAX_JUNCTION_C:
            raise ThermalRunaway(
                f"pass {n}: the junction reached {hottest:.1f} degC, past the "
                f"{MAX_JUNCTION_C:.0f} degC where the device model stops describing "
                "silicon. The loop is amplifying; there is no steady state to report")
        if previous_delta is not None and delta > previous_delta:
            growth_streak += 1
        else:
            growth_streak = 0
        if growth_streak >= 2 and n >= 3:
            raise ThermalRunaway(
                f"pass {n}: the junction moved {delta:.2f} K after two passes of "
                "growing moves. The loop is diverging, so no number it produces is a "
                "steady state")
        previous_delta = delta

        if delta <= tol_k:
            converged = True
            break

        guess = _extrapolated(got, signed, previous_signed) if accelerate else dict(got)
        previous_signed = signed

    # The loop gain at the point it landed on, per sourced chiplet: watts per
    # kelvin from the device, times kelvin per watt from the solve. The thermal
    # resistance here is the part's own rise over its own power, so it ignores
    # the heat its neighbours put into it -- an approximation that under-states
    # the coupling on a crowded package and is named rather than hidden.
    gain: Dict[str, dict] = {}
    for s in sources:
        t = solved_junction.get(s.chiplet_id, first_guess)
        p = float(s.power_w(t))
        dpdt = _dp_dt(s, t)
        r_th = (t - ambient_c) / p if p > 0.0 else math.inf
        g = dpdt * r_th if math.isfinite(r_th) else math.inf
        gain[s.chiplet_id] = {
            "dp_dt_w_k": float(dpdt),
            "r_th_k_w": float(r_th),
            "loop_gain": float(g),
            "stable": bool(math.isfinite(g) and g < 1.0),
        }

    unstable = [i for i, g in gain.items() if not g["stable"]]
    if unstable and not converged:
        raise ThermalRunaway(
            "the loop did not settle in "
            f"{max_passes} passes and its gain is at or above 1 on "
            + ", ".join(sorted(unstable))
            + ": that is a runaway, not a tolerance to loosen")

    # How many passes this loop needs, from the two moves it actually made.
    # A linear iteration closes a fixed fraction each pass, so the remaining
    # moves are a geometric series and the count follows. Reported rather than
    # acted on: the budget is the caller's to set, but a caller that did not
    # settle deserves to know whether it missed by one pass or by six.
    needed: Optional[int] = None
    if not converged and len(passes) >= 2:
        d_now = passes[-1]["delta_k"]
        d_prev = passes[-2]["delta_k"]
        if 0.0 < d_now < d_prev and d_now > tol_k:
            ratio = d_now / d_prev
            extra = math.log(tol_k / d_now) / math.log(ratio)
            needed = len(passes) + int(math.ceil(extra))

    resolution = payload.get("resolution") or {}
    return {
        "ok": True,
        "converged": converged,
        "passes_needed_estimate": needed,
        "accelerated": bool(accelerate),
        "passes": passes,
        "pass_count": len(passes),
        "tol_k": float(tol_k),
        "max_passes": int(max_passes),
        "junction_c": dict(solved_junction),
        "power_w": _powers_at(sources, solved_junction, first_guess),
        "loop_gain": gain,
        "solve": payload,
        "sources": {s.chiplet_id: s.describe() for s in sources},
        "hotspots_resolved": resolution.get("ok"),
        "label": ("verified solve, with a power source that is modelled rather than "
                  "measured: the package physics carries its own validation label, "
                  "the device I-V carries none"),
    }


def summary_lines(out: dict, indent: str = "  ") -> List[str]:
    """The coupled result as terminal lines, for a report or a console run."""
    lines: List[str] = []
    settled = "settled" if out["converged"] else "DID NOT SETTLE"
    lines.append(f"{indent}Electro-thermal loop: {settled} in {out['pass_count']} "
                 f"pass(es), tolerance {out['tol_k']:.3g} K")
    if not out["converged"]:
        need = out.get("passes_needed_estimate")
        extra = f"about {need} passes would" if need else "more passes may"
        lines.append(f"{indent}  the loop is still contracting, so this is a budget "
                     f"that ran out rather than a runaway: {extra} reach the "
                     "tolerance")
    for p in out["passes"]:
        total = p["total_power_w"]
        lines.append(f"{indent}  pass {p['pass']}: {total:8.3f} W -> peak "
                     f"{p['peak_c']:7.2f} degC, junction moved {p['delta_k']:7.3f} K")
    for cid, g in sorted(out["loop_gain"].items()):
        verdict = "stable" if g["stable"] else "UNSTABLE"
        lines.append(f"{indent}  {cid}: dP/dT {g['dp_dt_w_k']:+.4g} W/K, "
                     f"Rth {g['r_th_k_w']:.4g} K/W, loop gain "
                     f"{g['loop_gain']:.4g} ({verdict})")
    if out.get("hotspots_resolved") is False:
        lines.append(f"{indent}  note: the solve failed its hotspots-resolved check, so "
                     "the junction fed back is a lower bound and the power is "
                     "under-stated")
    lines.append(f"{indent}  {out['label']}")
    return lines
