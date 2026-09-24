"""
tools/ai_train_language.py
RISC3D-AI :: train the copilot to read plain English, and measure it honestly.

    python tools/ai_train_language.py --register

Trains on generated paraphrases (ai/language_corpus.py) and measures on
validation/language_heldout.json -- requests written by hand, in a different
voice, none of them produced by the generator, including typos and requests
the app cannot do at all. Training accuracy on generated text would only say
the model memorised the templates.

The abstention threshold is chosen on a GENERATED validation split for a
target precision, never on the held-out set: a threshold tuned on the set you
report is a threshold that has seen the answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import language as L  # noqa: E402
from ai import language_corpus as C  # noqa: E402
from ai import registry  # noqa: E402

HELD_OUT = ROOT / "validation" / "language_heldout.json"
HELD_OUT_B = ROOT / "validation" / "language_heldout_b.json"
OUT = ROOT / "validation" / "s4_language.json"


def featurise(rows: Sequence[Tuple[str, str]], vocab, dim: int):
    x = np.zeros((len(rows), dim))
    y = np.zeros(len(rows), dtype=int)
    index = {c: i for i, c in enumerate(C.INTENTS)}
    for n, (text, intent) in enumerate(rows):
        masked, _ids = L.mask(text, vocab)
        x[n] = L.features(masked, dim)
        y[n] = index[intent]
    return x, y


def train(x, y, classes: int, epochs: int = 60, lr: float = 0.5, batch: int = 256,
          l2: float = 1e-6, seed: int = 0, xv=None, yv=None):
    rng = np.random.default_rng(seed)
    w = np.zeros((x.shape[1], classes))
    b = np.zeros(classes)
    mw, vw = np.zeros_like(w), np.zeros_like(w)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    best, best_state, stale = -1.0, None, 0
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        for s in range(0, len(x), batch):
            idx = order[s:s + batch]
            xb, yb = x[idx], y[idx]
            z = xb @ w + b
            z -= z.max(axis=1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(axis=1, keepdims=True)
            p[np.arange(len(idx)), yb] -= 1.0
            p /= len(idx)
            gw = xb.T @ p + l2 * w
            gb = p.sum(axis=0)
            t += 1
            for param, grad, m, v in ((w, gw, mw, vw), (b, gb, mb, vb)):
                m *= 0.9
                m += 0.1 * grad
                v *= 0.999
                v += 0.001 * grad * grad
                param -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if xv is not None:
            acc = float((np.argmax(xv @ w + b, axis=1) == yv).mean())
            if acc > best + 1e-5:
                best, best_state, stale = acc, (w.copy(), b.copy()), 0
            else:
                stale += 1
                if stale >= 8:
                    break
    if best_state is not None:
        w, b = best_state
    return w, b, best


def choose_threshold_on_requests(model: L.LanguageModel, n: int = 400, seed: int = 4242
                                 ) -> Tuple[float, Dict[float, float]]:
    """Pick the abstention threshold on generated WHOLE REQUESTS, scored the
    way the copilot is scored: read the request, compare the set of intents.

    A per-clause threshold picked on per-clause accuracy answers the wrong
    question -- the pipeline splits, merges and abstains, and those interact.
    This runs the real pipeline at each threshold and takes the best F1. It is
    still generated text: the hand-written sets are never consulted, or they
    would stop being held out.
    """
    import random as _random
    rng = _random.Random(seed)
    names = ("CPU", "GPU", "memory", "SerDes", "NPU")
    parts = [{"id": f"p{i}", "name": name} for i, name in enumerate(names)]
    cases = []
    for _ in range(n):
        text, labelled = C.request(rng, names)
        cases.append((text, {intent for _piece, intent in labelled if intent != "NONE"}))
    # Requests made only of things the app cannot do. A threshold that reads
    # an instruction into one of these is not a threshold worth having, whatever
    # it does for F1, so it is a constraint rather than another term to trade.
    junk = []
    for _ in range(n // 3):
        pieces = [C.clause(rng, "NONE", names) for _ in range(rng.randint(1, 3))]
        junk.append(", ".join(pieces))
    scores: Dict[float, float] = {}
    quiet: Dict[float, float] = {}
    for threshold in [round(t, 2) for t in np.arange(0.15, 0.85, 0.05)]:
        model.threshold = float(threshold)
        tp = fp = fn = 0
        for text, want in cases:
            got, _ = read_request(text, parts, model)
            have = set(got)
            tp += len(want & have)
            fp += len(have - want)
            fn += len(want - have)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        clean = sum(not read_request(text, parts, model)[0] for text in junk) / len(junk)
        scores[threshold] = (2 * precision * recall / (precision + recall)
                             if precision + recall else 0.0)
        quiet[threshold] = clean
    allowed = [t for t in scores if quiet[t] >= 0.95] or list(scores)
    best = max(allowed, key=lambda t: scores[t])
    model.threshold = best
    return best, scores


def choose_threshold(model: L.LanguageModel, x, y, target_precision: float = 0.97) -> float:
    """The lowest threshold that still gets `target_precision` on the clauses
    it accepts. Chosen on GENERATED validation text, never on the held-out set."""
    probs = model.scores(x)
    top = probs.max(axis=1)
    pred = np.argmax(probs, axis=1)
    none_class = list(C.INTENTS).index("NONE")
    act = pred != none_class                       # clauses the model would act on
    # A floor, because a generated validation set is easier than real text:
    # at 100 % accuracy every threshold looks perfectly precise, and the
    # lowest one would accept every fragment a real sentence leaves behind.
    best = 0.95
    for threshold in np.arange(0.40, 0.96, 0.01):
        take = act & (top >= threshold)
        if take.sum() == 0:
            continue
        precision = float((pred[take] == y[take]).mean())
        if precision >= target_precision:
            best = float(threshold)
            break
    return best


def read_request(text: str, parts: List[dict], model: L.LanguageModel):
    got: Dict[str, List[str]] = {}
    misses = []
    for r in L.read(text, parts, model):
        if r.understood:
            got.setdefault(r.intent, [])
            for pid in r.parts:
                if pid not in got[r.intent]:
                    got[r.intent].append(pid)
        elif r.intent != "NONE" or r.confidence < model.threshold:
            misses.append(r.text)
    return got, misses


def evaluate(model: L.LanguageModel, path: Path = HELD_OUT) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    parts = [{"id": pid, "name": name} for pid, name in data["design"].items()]
    tp = fp = fn = 0
    exact = junk_ok = junk = 0
    part_right = part_total = 0
    wrong: List[Dict] = []
    for case in data["requests"]:
        want = set(case["intents"])
        got, _misses = read_request(case["text"], parts, model)
        have = set(got)
        tp += len(want & have)
        fp += len(have - want)
        fn += len(want - have)
        if want == have:
            exact += 1
        else:
            wrong.append({"text": case["text"], "wanted": sorted(want), "got": sorted(have)})
        if not want:
            junk += 1
            junk_ok += int(not have)
        for intent, expected in (case.get("parts") or {}).items():
            if intent in got:
                part_total += 1
                part_right += int(set(got[intent]) == set(expected))
    n = len(data["requests"])
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"requests": n, "exact_request_match_pct": 100.0 * exact / n,
            "intent_precision": precision, "intent_recall": recall,
            "intent_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "junk_requests": junk, "junk_ignored_pct": 100.0 * junk_ok / junk if junk else 100.0,
            "part_linking_pct": 100.0 * part_right / part_total if part_total else 100.0,
            "mistakes": wrong}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clauses", type=int, default=40000)
    ap.add_argument("--dim", type=int, default=L.DIM)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--precision", type=float, default=0.97,
                    help="target precision on accepted clauses, for the abstention threshold")
    ap.add_argument("--register", action="store_true")
    args = ap.parse_args()

    train_rows = C.corpus(args.clauses, seed=0)
    val_rows = C.corpus(args.clauses // 5, seed=99)      # different draw, same generator
    vocab = L.part_vocabulary([{"id": p, "name": p} for p in
                               ("CPU", "GPU", "memory", "SerDes", "NPU", "HBM",
                                "accelerator", "io die", "cache", "the cpu core")])
    t0 = time.perf_counter()
    xt, yt = featurise(train_rows, vocab, args.dim)
    xv, yv = featurise(val_rows, vocab, args.dim)
    w, b, val_acc = train(xt, yt, len(C.INTENTS), epochs=args.epochs, xv=xv, yv=yv)
    model = L.LanguageModel(weights=w, bias=b, classes=tuple(C.INTENTS), dim=args.dim)
    threshold, curve = choose_threshold_on_requests(model)
    train_s = time.perf_counter() - t0
    print(f"trained on {len(train_rows):,} generated clauses in {train_s:.0f} s")
    print(f"  generated validation accuracy {val_acc * 100:.1f} %")
    print(f"  abstention threshold {threshold:.2f}: best F1 ({curve[threshold]:.3f}) among the "
          f"settings that leave 95 % of generated junk requests alone")

    held = evaluate(model, HELD_OUT)
    held_b = evaluate(model, HELD_OUT_B) if HELD_OUT_B.exists() else None
    for label, result in (("set A (informed the category fixes)", held),
                          ("set B (written after them, measured once)", held_b)):
        if result is None:
            continue
        print(f"\nHAND-WRITTEN held-out {label} -- {result['requests']} requests, "
              f"none from the generator:")
        print(f"  whole request read exactly right : {result['exact_request_match_pct']:.0f} %")
        print(f"  intent precision / recall / F1   : {result['intent_precision']:.2f} / "
              f"{result['intent_recall']:.2f} / {result['intent_f1']:.2f}")
        print(f"  parts linked to the right ask    : {result['part_linking_pct']:.0f} %")
        print(f"  requests it cannot do, ignored   : {result['junk_ignored_pct']:.0f} % "
              f"of {result['junk_requests']}")
        if result["mistakes"]:
            print(f"  {len(result['mistakes'])} read wrong:")
            for m in result["mistakes"][:10]:
                print(f"    {m['text'][:62]:64s} wanted {m['wanted']} got {m['got']}")

    card = {"kind": "hashed n-gram softmax over masked clauses",
            "training": {"clauses": len(train_rows), "dim": args.dim, "epochs": args.epochs,
                         "generated_val_accuracy": val_acc, "threshold": model.threshold,
                         "threshold_chosen_on": "generated requests, best F1",
                         "threshold_curve": curve, "train_s": train_s},
            "held_out": {k: v for k, v in held.items() if k != "mistakes"},
            "held_out_b": None if held_b is None else {k: v for k, v in held_b.items()
                                                       if k != "mistakes"},
            "note": ("decides what to ask the solver; never produces a temperature. "
                     "Measured on hand-written requests, not on generated text.")}
    OUT.write_text(json.dumps({**card, "mistakes": held["mistakes"],
                               "mistakes_b": None if held_b is None else held_b["mistakes"]},
                              indent=2, default=float), encoding="utf-8")
    if args.register:
        version, path = registry.register("copilot-language", model.to_json(), card)
        registry.log_run("train-language", card["training"], card["held_out"],
                         model=f"copilot-language/v{version}")
        print(f"\nregistered copilot-language v{version}: {path}")
    else:
        registry.log_run("train-language (not registered)", card["training"], card["held_out"])
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
