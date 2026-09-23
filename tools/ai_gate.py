"""
tools/ai_gate.py
RISC3D-AI :: the S4 gate, measured.

    Surrogate error at the 95th percentile sits inside the S1 error bar on
    held-out packages; guided search scores at least 5x fewer layouts and
    returns the identical optimum on a regression set; the copilot is
    structurally unable to state a temperature that is not in a solve record.

    python tools/ai_gate.py

Writes validation/s4_gate.json. Each part is measured, not asserted:

  surrogate  the registered model against the TEST packages of its dataset --
             packages it never trained or calibrated on -- judged against each
             design's own error band, the band S1 confirmed against measured
             silicon.
  search     the shipped search and the guided one on the same instances, from
             the same starting layouts. The optima must be identical; the
             layouts scored are compared per instance and in total. The split
             between what the exact prior-cell screen does and what the LEARNED
             prior adds is reported, because only one of them is learning.
  copilot    real questions, audited output, and an adversarial explainer that
             tries to state a temperature it cannot cite -- which must be
             refused rather than printed.
"""

from __future__ import annotations

import json
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import studio.engine as E  # noqa: E402
from ai import copilot as C  # noqa: E402
from ai import dataset as D  # noqa: E402
from ai import guided as G  # noqa: E402
from ai import records as rec  # noqa: E402
from ai import rom as R  # noqa: E402
from ai import surrogate as S  # noqa: E402
from studio.scenarios import scenarios  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT = ROOT / "validation" / "s4_gate.json"
TARGET_RATIO = 5.0


# ---------------------------------------------------------------------------
# 1. the surrogate, on packages it has never seen
# ---------------------------------------------------------------------------

def surrogate_gate() -> dict:
    model, card = S.load_registered()
    ds = D.load(card["dataset"]["id"].split("/")[0], int(card["dataset"]["id"].split("/v")[1]))
    if ds.manifest["labels_sha256"] != card["dataset"]["sha256"]:
        raise SystemExit("the registered model names a different version of its dataset")
    test = ds.part("test")
    cache = R.RomCache(capacity=8, factor=model.rom_factor)
    X, _y, owner, designs = S.rows_from_labels(test, cache)
    peaks = S.design_predictions(model, X, owner, designs)
    m = S.metrics(peaks, designs)
    rom = S.metrics(S.design_predictions(None, X, owner, designs), designs)
    cov = S.coverage(model, peaks, designs)
    return {"model": f"thermal-surrogate/v{card['version']}", "dataset": card["dataset"]["id"],
            "test_designs": m["designs"], "test_packages": len(ds.manifest["split"]["test"]),
            "p95_err_over_band": m["err_over_band"]["p95"], "inside_band_pct": m["inside_band_pct"],
            "abs_err_k": m["abs_err_k"], "rel_err_pct": m["rel_err_pct"],
            "rom_only_p95_err_over_band": rom["err_over_band"]["p95"],
            "conformal_coverage_pct": cov * 100, "predict_ms": card["metrics"]["predict_ms"],
            "passed": bool(m["err_over_band"]["p95"] <= 1.0)}


# ---------------------------------------------------------------------------
# 2. the guided search, against the shipped one
# ---------------------------------------------------------------------------

def regression_set():
    for sc in scenarios():
        yield sc["id"], sc["package"], sc["chiplets"]
    rng = np.random.default_rng(4242)
    patterns = tuple(E.POWER_PATTERNS)
    for t, (size, pitch, n) in enumerate([(20.0, 2.0, 4), (20.0, 2.0, 5), (20.0, 1.0, 4),
                                          (30.0, 2.0, 5), (40.0, 4.0, 6), (40.0, 4.0, 7),
                                          (10.0, 1.0, 3), (30.0, 2.0, 4)]):
        parts, taken = [], []
        sides = (2.0, 3.0, 4.0, 5.0) if size <= 20 else (4.0, 6.0, 8.0)
        while len(parts) < n:
            w, h = float(rng.choice(sides)), float(rng.choice(sides))
            if w > size - 2 or h > size - 2:
                continue
            x = float(rng.integers(0, int((size - w) // pitch) + 1)) * pitch
            y = float(rng.integers(0, int((size - h) // pitch) + 1)) * pitch
            if any(x < ox + ow + 1 and ox < x + w + 1 and y < oy + oh + 1 and oy < y + h + 1
                   for ox, oy, ow, oh in taken):
                continue
            taken.append((x, y, w, h))
            k = len(parts)
            parts.append({"id": f"c{k}", "name": f"C{k}", "w_mm": w, "h_mm": h,
                          "power_w": float(np.round(rng.uniform(1, 10), 2)),
                          "power_pattern": str(rng.choice(patterns)), "x_mm": x, "y_mm": y})
        yield (f"random-{size:.0f}mm-{t}", {"size_mm": size, "h_w_m2k": float(rng.choice(
            [1000, 2500, 5000, 25000])), "ambient_c": 25.0, "pitch_mm": pitch}, parts)


def search_gate() -> dict:
    cache = E.ModelCache(capacity=2)
    rows = []
    for name, raw_pkg, raw_parts in regression_set():
        spec = E.PackageSpec.parse(raw_pkg)
        chips = E.parse_chiplets(raw_parts, spec)
        t0 = time.perf_counter()
        base = E.optimize(cache, spec, chips)["certificate"]
        base_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        guided = E.optimize(cache, spec, chips, search="guided")["certificate"]
        guided_s = time.perf_counter() - t0
        same = abs(base["lattice_optimum_c"] - guided["lattice_optimum_c"]) < 1e-9
        rows.append({"case": name, "combinations": base["combinations"],
                     "shipped_scored": base["layouts_scored"],
                     "guided_scored": guided["layouts_scored"],
                     "guided_prescreened": guided["layouts_prescreened"],
                     "ratio": base["layouts_scored"] / max(guided["layouts_scored"], 1),
                     "identical_optimum": bool(same),
                     "shipped_optimum_c": base["lattice_optimum_c"],
                     "guided_optimum_c": guided["lattice_optimum_c"],
                     "shipped_s": base_s, "guided_s": guided_s})
        print(f"  {name:20s} scored {base['layouts_scored']:>9,} -> {guided['layouts_scored']:>7,}"
              f"  ({rows[-1]['ratio']:6.1f}x)  {'same optimum' if same else 'DIFFERENT OPTIMUM'}",
              flush=True)
    total_ratio = sum(r["shipped_scored"] for r in rows) / max(
        sum(r["guided_scored"] for r in rows), 1)
    median_ratio = statistics.median(r["ratio"] for r in rows)
    identical = all(r["identical_optimum"] for r in rows)
    return {"cases": len(rows), "rows": rows, "total_ratio": total_ratio,
            "median_ratio": median_ratio, "identical_optimum": identical,
            "target_ratio": TARGET_RATIO,
            "passed": bool(identical and median_ratio >= TARGET_RATIO
                           and total_ratio >= TARGET_RATIO)}


def learned_share() -> dict:
    """How much of the screen is the analytic rule and how much is learned."""
    captured = {}
    real = E.branch_and_bound

    def spy(fields, positions, extents, signatures, incumbent, **kw):
        captured.update(fields=fields, positions=positions, extents=extents,
                        signatures=signatures, incumbent=incumbent, kw=kw)
        return real(fields, positions, extents, signatures, incumbent, **kw)

    E.branch_and_bound = spy
    cache = E.ModelCache(capacity=2)
    rows = []
    try:
        for name, raw_pkg, raw_parts in regression_set():
            spec = E.PackageSpec.parse(raw_pkg)
            chips = E.parse_chiplets(raw_parts, spec)
            base = E.optimize(cache, spec, chips)["certificate"]
            c = dict(captured)
            kw = {"bound_cells": c["kw"].get("bound_cells"),
                  "clearance_cells": c["kw"].get("clearance_cells", 0)}
            out = {}
            for label, prior in (("analytic", 0), ("learned", 4)):
                v, _ch, st = G.guided_branch_and_bound(
                    c["fields"], c["positions"], c["extents"], c["signatures"], c["incumbent"],
                    prescreen=True, prior_cells=prior, **kw)
                out[label] = st.leaves
            rows.append({"case": name, "shipped": base["layouts_scored"],
                         "analytic_only": out["analytic"], "with_learned_prior": out["learned"]})
    finally:
        E.branch_and_bound = real
    tot_ship = sum(r["shipped"] for r in rows)
    tot_an = sum(r["analytic_only"] for r in rows)
    tot_le = sum(r["with_learned_prior"] for r in rows)
    return {"rows": rows, "shipped_total": tot_ship, "analytic_total": tot_an,
            "learned_total": tot_le, "analytic_ratio": tot_ship / max(tot_an, 1),
            "learned_extra_ratio": tot_an / max(tot_le, 1),
            "combined_ratio": tot_ship / max(tot_le, 1)}


# ---------------------------------------------------------------------------
# 3. the copilot: it cannot state a temperature it cannot cite
# ---------------------------------------------------------------------------

class Liar(C.Explainer):
    """An explainer that states a temperature no record holds."""

    def explain(self, store, ctx):
        return "The layout will run at 42.0 °C and everything is fine."


class Forger(C.Explainer):
    """An explainer that cites a real record but prints a different number."""

    def explain(self, store, ctx):
        r = ctx["runs"][0][1]
        return f"The proven optimum is 1.0 °C [rec:{r.id}:optimized.peak_c]."


def copilot_gate() -> dict:
    sc = scenarios()[0]
    cache = E.ModelCache(capacity=2)
    questions = ["under 80 C at 12 W, memory on the left, SerDes at the edge",
                 "keep the NPU in the centre and find the coolest layout",
                 "under 40 C with a cold plate, CPU near memory",
                 "make it purple and quiet"]
    checks = []
    for q in questions:
        store = rec.RecordStore()
        a = C.answer(q, sc["package"], sc["chiplets"], store, cache)
        cites = rec.audit(a.narrative, store)          # raises if anything is uncited
        checks.append({"question": q, "citations": len(cites),
                       "records": len(a.records), "not_understood": a.plan.not_understood,
                       "meets_limit": a.meets_limit,
                       "every_temperature_cited": True})
    refused = {}
    for label, explainer in (("uncited temperature", Liar()), ("cited but wrong value", Forger())):
        store = rec.RecordStore()
        a = C.answer(questions[0], sc["package"], sc["chiplets"], store, cache, explainer=explainer)
        rec.audit(a.narrative, store)
        refused[label] = {"fell_back_to_template": "42.0" not in a.narrative and "1.0 °C" not in a.narrative,
                          "narrative_audits": True}
    source = (ROOT / "ai" / "copilot.py").read_text(encoding="utf-8")
    offline = not any(mod in source for mod in ("import socket", "import urllib", "import requests",
                                                "http.client"))
    return {"questions": checks, "adversarial": refused, "offline": offline,
            "passed": bool(all(c["every_temperature_cited"] for c in checks)
                           and all(v["fell_back_to_template"] for v in refused.values())
                           and offline)}


def main() -> None:
    print("S4 gate\n\n1. surrogate on held-out packages")
    sur = surrogate_gate()
    print(f"   p95 |error| / band = {sur['p95_err_over_band']:.2f} (<= 1), inside the band on "
          f"{sur['inside_band_pct']:.0f} % of {sur['test_designs']} designs across "
          f"{sur['test_packages']} unseen packages: {'PASS' if sur['passed'] else 'FAIL'}")
    print("\n2. guided search against the shipped search")
    search = search_gate()
    share = learned_share()
    print(f"   layouts scored: {share['shipped_total']:,} -> {share['learned_total']:,} "
          f"(total {search['total_ratio']:.1f}x, median {search['median_ratio']:.1f}x); "
          f"identical optimum on all {search['cases']}: {'PASS' if search['passed'] else 'FAIL'}")
    print(f"   of which: exact prior-cell screen {share['analytic_ratio']:.1f}x, "
          f"the learned prior a further {share['learned_extra_ratio']:.2f}x")
    print("\n3. copilot")
    cop = copilot_gate()
    print(f"   {len(cop['questions'])} questions answered with every temperature cited; "
          f"an explainer that invents one is refused; offline: {'PASS' if cop['passed'] else 'FAIL'}")
    report = {"gate": ("surrogate p95 error inside the S1 band on held-out packages; guided search "
                       "scores at least 5x fewer layouts and returns the identical optimum; the "
                       "copilot cannot state a temperature that is not in a solve record"),
              "surrogate": sur, "search": {**search, "learned_share": share}, "copilot": cop,
              "passed": bool(sur["passed"] and search["passed"] and cop["passed"])}
    OUT.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nS4 GATE: {'PASSED' if report['passed'] else 'NOT MET'}")
    print(f"written: {OUT}")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
