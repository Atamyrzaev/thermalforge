"""
tools/ai_train.py
ThermalForge :: train the surrogate and measure it on held-out packages (S4 4.2).

    python tools/ai_train.py --dataset thermal-peaks --version 1

Trains ai/surrogate.py on the TRAIN packages, calibrates its conformal
interval on the CALIB packages, and measures it on the TEST packages -- which
it has never seen, because the split is by package. The reduced-order model
alone is measured on the same designs, at the surrogate's cell size and at the
finer one, so what the learning adds is visible rather than asserted.

Registers the model (data/models/thermal-surrogate/vN) with a card naming the
dataset and its sha256, and logs the run to data/experiments.jsonl.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import dataset as D  # noqa: E402
from ai import registry  # noqa: E402
from ai import rom as R  # noqa: E402
from ai import surrogate as S  # noqa: E402

for name in ("ThermalForge.ThermalSolver", "ThermalForge.GridBuilder", "ThermalForge.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="thermal-peaks")
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--rom-factor", type=float, default=2.0)
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--register", action="store_true", help="write the model to the registry")
    args = ap.parse_args()

    ds = D.load(args.dataset, args.version)
    train, calib, test = ds.part("train"), ds.part("calib"), ds.part("test")
    print(f"{ds.id}: {len(train)} train / {len(calib)} calib / {len(test)} test designs "
          f"(clean only), split by package")

    t0 = time.perf_counter()
    model, info = S.fit(train, calib, rom_factor=args.rom_factor, members=args.members,
                        seed=args.seed, hidden=args.hidden)
    train_s = time.perf_counter() - t0
    print(f"trained in {train_s:.0f} s: {info['train_rows']} part rows, conformal q95 "
          f"{model.q95_rel * 100:.1f} % of the predicted rise")

    cache = R.RomCache(capacity=8, factor=args.rom_factor)
    # Configurations are chosen on CALIBRATION packages, never on the test
    # split: selecting on the test set is how a held-out number stops meaning
    # anything. The test split is measured once, for the configuration chosen.
    Xc, _yc, oc, dc = S.rows_from_labels(calib, cache)
    calib_metrics = S.metrics(S.design_predictions(model, Xc, oc, dc), dc)
    print(f"calibration packages (used for choosing): p95 |err|/band "
          f"{calib_metrics['err_over_band']['p95']:.2f}, |err| p95 "
          f"{calib_metrics['abs_err_k']['p95']:.2f} K")
    X, _y, owner, designs = S.rows_from_labels(test, cache)
    rom_only = S.metrics(S.design_predictions(None, X, owner, designs), designs)
    peaks = S.design_predictions(model, X, owner, designs)
    learned = S.metrics(peaks, designs)
    # end-to-end prediction cost, cold ROM cache included every 8th design
    import studio.engine as E
    timing = R.RomCache(capacity=8, factor=args.rom_factor)
    t0 = time.perf_counter()
    for lab in test[:40]:
        spec = E.PackageSpec.parse(dict(lab["package"], mesh="standard"))
        model.predict(timing, spec, E.parse_chiplets(lab["chiplets"], spec))
    predict_s = (time.perf_counter() - t0) / max(1, len(test[:40]))
    cov = S.coverage(model, peaks, designs)

    fine = R.RomCache(capacity=8, factor=1.0)
    Xf, _yf, ownerf, designsf = S.rows_from_labels(test, fine)
    rom_fine = S.metrics(S.design_predictions(None, Xf, ownerf, designsf), designsf)

    print(f"\n{'held-out test packages':34s}{'|err| p50':>11}{'p95':>8}{'max':>8}"
          f"{'|err|/band p95':>16}{'inside band':>13}")
    for label, m in (("ROM only, surrogate cell", rom_only), ("ROM only, engine cell", rom_fine),
                     ("ROM + learned correction", learned)):
        print(f"{label:34s}{m['abs_err_k']['p50']:>10.2f}K{m['abs_err_k']['p95']:>7.2f}K"
              f"{m['abs_err_k']['max']:>7.2f}K{m['err_over_band']['p95']:>16.2f}"
              f"{m['inside_band_pct']:>12.0f}%")
    print(f"\nconformal 95 % interval covers {cov * 100:.1f} % of held-out designs "
          f"(target 95 %); prediction {predict_s * 1e3:.1f} ms per design")
    gate = learned["err_over_band"]["p95"] <= 1.0
    print(f"S4 surrogate gate -- p95 of |error|/band on held-out packages "
          f"{learned['err_over_band']['p95']:.2f} <= 1: {'PASS' if gate else 'FAIL'}")

    card = {"kind": "rom+correction surrogate", "dataset": {"id": ds.id, "sha256": ds.manifest["labels_sha256"]},
            "training": {"rom_factor": args.rom_factor, "members": args.members, "hidden": args.hidden,
                         "seed": args.seed, "train_designs": len(train), "calib_designs": len(calib),
                         "train_s": train_s, **info},
            "metrics": {"test_designs": len(test), "learned": learned, "calib": calib_metrics,
                        "rom_only": rom_only,
                        "rom_only_engine_cell": rom_fine, "coverage": cov,
                        "predict_ms": predict_s * 1e3},
            "gate": {"p95_err_over_band": learned["err_over_band"]["p95"], "passed": bool(gate)},
            "note": "never produces a reported number; ranks, screens and guides only"}
    if args.register:
        version, path = registry.register("thermal-surrogate", model.to_json(), card)
        print(f"registered thermal-surrogate v{version}: {path}")
        registry.log_run("train", card["training"], card["metrics"],
                         dataset={"id": ds.id, "sha256": ds.manifest["labels_sha256"]},
                         model=f"thermal-surrogate/v{version}")
    else:
        registry.log_run("train (not registered)", card["training"], card["metrics"],
                         dataset={"id": ds.id, "sha256": ds.manifest["labels_sha256"]})
    (ROOT / "validation" / "s4_surrogate.json").write_text(json.dumps(card, indent=2, default=float),
                                                           encoding="utf-8")


if __name__ == "__main__":
    main()
