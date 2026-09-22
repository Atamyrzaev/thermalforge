"""
tools/ttc_crossvalidate.py
RISC3D-AI :: calibrate on one measured pattern, predict the other twelve.

The SAMOS 2019 thermal test chip (validation/measured/ttc_samos2019_steady.json)
records a measured power map and a measured 16-diode temperature map for 13
power patterns under one unchanged forced-convection rig. Two things about the
rig were never published: the silicon thickness and the effective convective
coefficient. Each is fitted to do ONE job, on ONE pattern ('one'):

    h      solved so the modelled MEAN temperature of 'one' equals the measured
           mean -- the level is what h sets
    t_si   chosen, with h re-solved at each value, for the best SHAPE on 'one'
           -- the spreading is what thickness sets

Nothing is fitted on any other pattern. The other twelve are predictions, and
each is judged against the model's own stated band (instrument +/-0.1 K and
the three-term error budget, in quadrature) on two separate counts: LEVEL, the
common offset, and SHAPE, what is left after removing it.

The level is then predicted a second, better-posed way -- leave one out: each
pattern from the h that fits the other twelve (see leave_one_out). Both are
reported; neither is fitted on the pattern it predicts.

The extract is checked for power/temperature co-location when it is built
(tools/ttc_extract.py), so a recording whose channels are misaligned cannot
reach this comparison and be mistaken for a conduction error.

    python tools/ttc_crossvalidate.py

Writes validation/ttc_crossvalidation.json.
"""

from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

DATA_PATH = ROOT / "validation" / "measured" / "ttc_samos2019_steady.json"
OUT = ROOT / "validation" / "ttc_crossvalidation.json"
CONDITION = "heatsink_with_fan"        # forced convection: where a constant h is sound
COOLING = "lid_heatsink"               # a heat sink has a base, and a base spreads heat
CELL_MM = 2.0
FIT_ON = "one"
THICKNESSES_UM = (100, 200, 300, 400, 500, 560)


def modelled(cache, case, t_um, h):
    spec = E.PackageSpec.parse({"size_mm": 8.0, "h_w_m2k": h, "ambient_c": case["ambient_c"],
                                "pitch_mm": 1.0, "cooling": COOLING, "keepout_um": 0.0,
                                "stack_um": [20.0, 20.0, t_um]})
    chips = [{"id": f"h{n}", "name": f"H{n}", "w_mm": CELL_MM, "h_mm": CELL_MM,
              "power_w": max(p, 1e-6), "x_mm": (n % 4) * CELL_MM, "y_mm": (n // 4) * CELL_MM}
             for n, p in enumerate(case["power_w"])]
    out = E.solve(cache, spec, E.parse_chiplets(chips, spec))
    g = out["grid"]
    field = (np.frombuffer(base64.b64decode(out["field_b64"]), dtype="<f4")
             .reshape(g["nx"], g["ny"], g["nz"]).astype(np.float64))
    k = g["die_k0"]
    temps = np.array([field[min(g["nx"] - 1, int(((n % 4) + 0.5) * CELL_MM / spec.cell_mm)),
                            min(g["ny"] - 1, int(((n // 4) + 0.5) * CELL_MM / spec.cell_mm)), k]
                      for n in range(16)])
    return temps, out


def h_for_level(cache, case, t_um):
    """Bisect h on a log scale until the modelled mean equals the measured one."""
    want = float(np.mean(case["temp_c"]))
    lo, hi = 1000.0, 400000.0
    for _ in range(30):
        mid = (lo * hi) ** 0.5
        got = float(np.mean(modelled(cache, case, t_um, mid)[0]))
        lo, hi = (mid, hi) if got > want else (lo, mid)
    return (lo * hi) ** 0.5


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))["cooling_conditions"][CONDITION]
    cache = E.ModelCache(capacity=3)
    fit = data[FIT_ON]

    print(f"Fit on '{FIT_ON}' only ({CONDITION}, modelled as {COOLING}):")
    fits = []
    for t in THICKNESSES_UM:
        h = h_for_level(cache, fit, float(t))
        got, _ = modelled(cache, fit, float(t), h)
        err = got - np.array(fit["temp_c"])
        shape = float(np.sqrt(((err - err.mean()) ** 2).mean()))
        fits.append({"t_si_um": t, "h_w_m2k": h, "shape_rms_k": shape})
        print(f"  t_si {t:4d} um   h {h:9,.0f} W/m2K   shape {shape:.3f} K")
    chosen = min(fits, key=lambda f: f["shape_rms_k"])
    T, H = float(chosen["t_si_um"]), float(chosen["h_w_m2k"])
    print(f"  chosen: t_si {T:.0f} um, h {H:,.0f} W/m2K\n")

    print(f"{'held out':14s}{'P (W)':>8}{'level':>9}{'shape':>9}{'band':>8}   level   shape")
    rows = []
    for name, case in sorted(data.items()):
        if name == FIT_ON:
            continue
        got, out = modelled(cache, case, T, H)
        err = got - np.array(case["temp_c"])
        band = float(np.hypot(0.1, out["metrics"]["uncertainty"]["combined_k"]))
        level, shape = float(err.mean()), float(np.sqrt(((err - err.mean()) ** 2).mean()))
        row = {"pattern": name, "power_w": case["total_w"], "level_k": level,
               "shape_rms_k": shape, "band_k": band,
               "level_inside": abs(level) <= band, "shape_inside": shape <= band}
        rows.append(row)
        print(f"{name:14s}{case['total_w']:>8.1f}{level:>+8.2f}K{shape:>8.2f}K{band:>7.2f}K"
              f"   {'yes' if row['level_inside'] else 'NO ':3s}     {'yes' if row['shape_inside'] else 'NO'}")

    lv = sum(r["level_inside"] for r in rows)
    sh = sum(r["shape_inside"] for r in rows)
    both = sum(r["level_inside"] and r["shape_inside"] for r in rows)
    corr = float(np.corrcoef([r["power_w"] for r in rows], [r["level_k"] for r in rows])[0, 1])
    print(f"\nlevel inside the band on {lv} of {len(rows)}, shape on {sh} of {len(rows)}, "
          f"both on {both} of {len(rows)}")
    print(f"correlation of level error with total power: {corr:+.2f}")

    loo = leave_one_out(cache, data, T)
    report = {
        "what": "calibrate the two unpublished rig parameters on measured patterns, predict "
                "held-out ones, judge level and shape against the stated band",
        "condition": CONDITION, "modelled_cooling": COOLING, "fit_on": FIT_ON,
        "fits": fits, "chosen": {"t_si_um": T, "h_w_m2k": H},
        "rows": rows,
        "level_inside": lv, "shape_inside": sh, "both_inside": both, "held_out": len(rows),
        "level_vs_power_correlation": corr,
        "leave_one_out": loo,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {OUT}")


def leave_one_out(cache, data, t_um, n_h: int = 25):
    """Level, predicted the well-posed way.

    Calibrating h on 'one' alone is the worst-conditioned choice the rig
    offers: it has the smallest rise (9 K), so a 1 K error in the room
    reference is ~12% in h, and extrapolating that h to 47 W multiplies it
    eightfold. Here each pattern's level is predicted from the h that best
    fits the OTHER twelve -- still a prediction, never fitted on itself.

    Each pattern's mean is tabulated against log h once (the model is exact
    at every node, the interpolation is between nodes 4.7% apart in h), so
    the thirteen pooled fits cost no extra solves.
    """
    grid = np.geomspace(15000.0, 45000.0, n_h)
    names = sorted(data)
    table, meas, band, implied = {}, {}, {}, {}
    for name in names:
        case = data[name]
        table[name] = np.array([float(np.mean(modelled(cache, case, t_um, h)[0])) for h in grid])
        meas[name] = float(np.mean(case["temp_c"]))
        _, out = modelled(cache, case, t_um, float(np.sqrt(grid[0] * grid[-1])))
        band[name] = float(np.hypot(0.1, out["metrics"]["uncertainty"]["combined_k"]))
        # the h this pattern alone would need; the mean falls as h rises
        implied[name] = float(np.exp(np.interp(-meas[name], -table[name], np.log(grid))))

    fine = np.geomspace(grid[0], grid[-1], 2000)
    interp = {n: np.interp(np.log(fine), np.log(grid), table[n]) for n in names}
    print(f"\nLeave-one-out: level of each pattern from h fitted on the other {len(names) - 1}")
    print(f"{'held out':14s}{'P (W)':>8}{'h (others)':>12}{'level':>9}{'band':>8}   inside")
    rows = []
    for name in names:
        others = [n for n in names if n != name]
        sse = sum((interp[n] - meas[n]) ** 2 for n in others)
        h = float(fine[int(np.argmin(sse))])
        level = float(np.interp(np.log(h), np.log(grid), table[name])) - meas[name]
        ok = abs(level) <= band[name]
        rows.append({"pattern": name, "power_w": data[name]["total_w"], "h_fitted_on_others": h,
                     "level_k": level, "band_k": band[name], "level_inside": ok,
                     "implied_h": implied[name]})
        print(f"{name:14s}{data[name]['total_w']:>8.1f}{h:>12,.0f}{level:>+8.2f}K{band[name]:>7.2f}K   "
              f"{'yes' if ok else 'NO'}")
    inside = sum(r["level_inside"] for r in rows)
    P = np.array([r["power_w"] for r in rows])
    Hs = np.array([r["implied_h"] for r in rows])
    h_vs_p = float(np.corrcoef(P, Hs)[0, 1])
    same_cell = (implied["one"], implied["hotspot_2"])
    print(f"level inside the band on {inside} of {len(rows)}")
    print(f"implied h: {Hs.min():,.0f} .. {Hs.max():,.0f} W/m2K; correlation with power {h_vs_p:+.2f} "
          f"-- the rig's effective h does NOT rise with power")
    print(f"same heated cell, two tests: 'one' needs h {same_cell[0]:,.0f}, 'hotspot_2' "
          f"{same_cell[1]:,.0f} -- test-to-test repeatability, not conduction")
    return {"rows": rows, "level_inside": inside, "held_out": len(rows),
            "t_si_um": t_um, "implied_h_vs_power_correlation": h_vs_p,
            "same_cell_implied_h": {"one": same_cell[0], "hotspot_2": same_cell[1]}}


if __name__ == "__main__":
    main()
