"""
tools/ai_language_compare.py
RISC3D-AI :: the phrase grammar and the trained reader, on the same requests.

    python tools/ai_language_compare.py

The copilot shipped first with a grammar: a list of phrasings, and "not
understood" for everything else. This reads the same hand-written held-out
requests with both, so the claim "it understands plain English now" is a
measurement rather than an impression.

Writes validation/s4_language_compare.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import copilot as CO  # noqa: E402
from ai import language as L  # noqa: E402

SETS = {"A": "language_heldout.json", "B": "language_heldout_b.json",
        "C": "language_heldout_c.json"}
KIND_INTENT = {v: k for k, v in CO.INTENT_KIND.items()}
COOLING_INTENT = {v: k for k, v in CO.INTENT_COOLING.items()}
OPTIMIZE = re.compile(r"\b(?:optimi[sz]e|place|arrange|coolest|layout|floorplan)\b", re.I)


def plan_intents(plan, text: str) -> Set[str]:
    """The asks a plan carries, as intent names, so both readers are scored
    the same way. The grammar has no field for 'optimise', so its own trigger
    words are counted for it -- scoring it on a gap it does not claim would
    not be a fair comparison."""
    got: Set[str] = set()
    if plan.max_peak_c is not None:
        got.add("LIMIT")
    if plan.total_power_w is not None:
        got.add("TOTAL_POWER")
    if plan.part_power_w:
        got.add("PART_POWER")
    if plan.size_mm is not None:
        got.add("SIZE")
    if plan.cooling:
        got.add(COOLING_INTENT[plan.cooling])
    if plan.near:
        got.add("NEAR")
    for c in plan.constraints:
        got.add(KIND_INTENT[c.kind])
    if OPTIMIZE.search(text):
        got.add("OPTIMIZE")
    return got


def score(reader, cases, parts) -> Dict:
    tp = fp = fn = exact = junk = junk_ok = 0
    for case in cases:
        want = set(case["intents"])
        got = reader(case["text"], parts)
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)
        exact += int(want == got)
        if not want:
            junk += 1
            junk_ok += int(not got)
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return {"requests": len(cases), "exact_pct": 100.0 * exact / len(cases),
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "junk_ignored_pct": 100.0 * junk_ok / junk if junk else 100.0}


def main() -> None:
    model, card = L.load_registered()
    out: Dict[str, Dict] = {"model": f"copilot-language/v{card['version']}", "sets": {}}
    print(f"{'set':4s}{'reader':10s}{'requests':>10}{'exact':>8}{'precision':>11}{'recall':>8}"
          f"{'F1':>7}{'junk ignored':>14}")
    for name, filename in SETS.items():
        data = json.loads((ROOT / "validation" / filename).read_text(encoding="utf-8"))
        parts = [{"id": pid, "name": label} for pid, label in data["design"].items()]
        cases = data["requests"]
        rows = {
            "grammar": score(lambda t, p: plan_intents(CO.parse_grammar(t, p), t), cases, parts),
            "trained": score(lambda t, p: plan_intents(CO.parse_learned(t, p, model), t), cases, parts),
        }
        out["sets"][name] = rows
        for reader, row in rows.items():
            print(f"{name:4s}{reader:10s}{row['requests']:>10}{row['exact_pct']:>7.0f}%"
                  f"{row['precision']:>11.2f}{row['recall']:>8.2f}{row['f1']:>7.2f}"
                  f"{row['junk_ignored_pct']:>13.0f}%")
    path = ROOT / "validation" / "s4_language_compare.json"
    path.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten: {path}")


if __name__ == "__main__":
    main()
