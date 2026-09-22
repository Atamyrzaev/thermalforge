"""
tools/ttc_extract.py
RISC3D-AI :: rebuild validation/measured/ttc_samos2019_steady.json from the raw
SAMOS 2019 thermal-test-chip traces.

The raw traces are not redistributed (38 MB); download and unpack them from
https://doi.org/10.5281/zenodo.2871796, then

    python tools/ttc_extract.py path/to/release

Each CSV row is one 10 Hz sample: 16 heater voltages, 16 heater currents, 16
diode temperatures (the dataset's own plot.sce reads them this way). The
steady state is the last stretch of the powered phase, before power is cut.

Before a recording is kept, the extract checks -- WITHOUT the model -- that the
temperatures belong to the cells the power went into: the measured power map
and the measured temperature map must correlate. That check is not a nicety.
Two fan recordings ('four', 'checkboard') have their 16 diode channels rotated
by one place: as recorded their power/temperature correlation is 0.24 and
-0.43, the hottest diodes sit one cell to the right of the powered heaters, and
the same two patterns are aligned under the other two cooling conditions.
Rotating the channels back by one restores 0.95 -- a clean checkerboard -- and
the channel that wraps around lands on an unpowered corner. The correction is
applied only when it is decisive (as recorded below 0.5, corrected above 0.9)
and is written into the extract with both correlations, so it can be audited
rather than trusted. A pattern whose symmetry makes a rotation a tie (a
checkerboard is unchanged by moving two rows) is left as recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "validation" / "measured" / "ttc_samos2019_steady.json"
CONDITIONS = ("chip", "heatsink", "heatsink_with_fan")
SAMPLE_HZ = 10.0
TAIL = (500, 100)             # samples before the end of the powered phase: 50 s .. 10 s
MISALIGNED_BELOW = 0.5
CORRECTED_ABOVE = 0.9


def ambients(folder: Path) -> dict:
    out = {}
    for line in (folder / "tests.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and "Tamb" in line:
            name, rest = line.split(None, 1)
            out[name] = float(rest.split("=")[1].replace("°C", "").strip())
    return out


def steady(path: Path):
    rows = np.loadtxt(path)
    P = rows[:, :16] * rows[:, 16:32]
    T = rows[:, 32:48]
    total = P.sum(axis=1)
    powered = np.flatnonzero(total > 0.8 * total.max())
    if powered.size < 300:
        return None
    tail = powered[-TAIL[0]:-TAIL[1]] if powered.size > 700 else powered[-200:]
    return P[tail].mean(axis=0), T[tail].mean(axis=0), T[tail].std(axis=0), total[tail].mean(), tail.size


def alignment(p: np.ndarray, t: np.ndarray):
    """Best cyclic rotation of the diode channels, judged by how well the
    temperature map follows the power map. None when power is too uniform
    for any rotation to be told apart."""
    if p.std() < 0.2:
        return None
    corr = [float(np.corrcoef(p, np.roll(t, -r))[0, 1]) for r in range(16)]
    best = int(np.argmax(corr))
    return {"as_recorded": corr[0], "best_rotation": best, "best": corr[best]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("release", type=Path, help="the unpacked dataset folder (contains chip/, heatsink/, ...)")
    args = ap.parse_args()

    data, corrections = {}, []
    for cond in CONDITIONS:
        folder = args.release / cond
        amb = ambients(folder)
        data[cond] = {}
        for csv in sorted(folder.glob("*.csv")):
            name = csv.stem
            s = steady(csv)
            if s is None:
                print(f"  {cond:18s}{name:12s} no steady powered phase -- skipped")
                continue
            p, t, sd, total, n = s
            a = alignment(p, t)
            note = "uniform power: alignment not identifiable" if a is None else f"corr {a['as_recorded']:+.3f}"
            if a is not None and a["as_recorded"] < MISALIGNED_BELOW:
                if a["best"] < CORRECTED_ABOVE:
                    print(f"  {cond}/{name}: misaligned and no rotation repairs it -- skipped")
                    continue
                r = a["best_rotation"]
                t, sd = np.roll(t, -r), np.roll(sd, -r)
                corrections.append({"recording": f"{cond}/{name}", "diode_rotation": r,
                                    "corr_as_recorded": a["as_recorded"], "corr_corrected": a["best"]})
                note = f"corr {a['as_recorded']:+.3f} -> {a['best']:+.3f} after rotating diodes by {r}"
            data[cond][name] = {"power_w": p.tolist(), "total_w": float(total), "temp_c": t.tolist(),
                                "temp_std_k": sd.tolist(), "samples": int(n), "ambient_c": amb.get(name)}
            print(f"  {cond:18s}{name:12s}{total:6.2f} W  T {t.min():5.1f}-{t.max():5.1f} C  {note}")

    report = {
        "source": ("F. Terraneo, A. Leva, W. Fornaciari, 'An Open-hardware Platform for MPSoC "
                   "Thermal Modeling', SAMOS 2019. Dataset: https://doi.org/10.5281/zenodo.2871796 "
                   "(CC-BY 4.0)."),
        "licence": "CC-BY 4.0 -- redistributed here as a derived extract with attribution",
        "what": (f"Steady-state averages extracted from the raw {SAMPLE_HZ:.0f} Hz traces by "
                 f"tools/ttc_extract.py: the last stretch of each powered phase, "
                 f"{TAIL[0] - TAIL[1]} samples ({(TAIL[0] - TAIL[1]) / SAMPLE_HZ:.0f} s). power_w is "
                 "MEASURED per heater (V x I), temp_c is the measured diode temperature per cell, "
                 "both in the dataset's cell order (row-major, top-left first, as in its "
                 "'experiments overview.txt'). The raw traces are not redistributed; they are at "
                 "the DOI above."),
        "geometry": ("4 x 4 heater cells, 2 x 2 mm each, 8 x 8 mm array -- pinned by the 3D-ICE 3.0 "
                     "paper's 5 W/cm2 uniform preheat against this dataset's 0.2 W baseline per "
                     "element."),
        "corrections": corrections,
        "corrections_rule": (f"a recording whose power/temperature correlation is below "
                             f"{MISALIGNED_BELOW} is misaligned; it is kept only if a cyclic rotation "
                             f"of its diode channels lifts the correlation above {CORRECTED_ABOVE}, "
                             "and the rotation is recorded here"),
        "cooling_conditions": data,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\n{len(corrections)} recording(s) corrected; written: {OUT}")


if __name__ == "__main__":
    sys.exit(main())
