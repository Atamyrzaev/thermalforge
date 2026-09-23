"""
tools/ai_dataset.py
RISC3D-AI :: build a versioned label set with the verified solver (S4 step 4.1).

    python tools/ai_dataset.py --name thermal-peaks --version 1 --packages 200 --layouts 20

Writes data/datasets/<name>/v<version>/ (manifest.json + labels.jsonl.gz) and
logs the run to data/experiments.jsonl. Versions are immutable: building one
that exists is refused. See ai/dataset.py for what a label carries and why
the split is by package.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai import dataset as D  # noqa: E402
from ai import registry  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="thermal-peaks")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--packages", type=int, default=200)
    ap.add_argument("--layouts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--mesh", default="standard")
    ap.add_argument("--workers", type=int, default=3,
                    help="processes; each holds up to two models (a fine one is ~0.6 GB)")
    args = ap.parse_args()

    def progress(i, raw, got):
        print(f"  package {i:4d}  {raw['size_mm']:4.0f} mm  {raw['cooling']:13s} "
              f"h {raw['h_w_m2k']:7.0f}  {got} labels", flush=True)

    result = D.generate(args.packages, args.layouts, args.seed, mesh=args.mesh,
                        workers=args.workers, on_package=progress)
    generator = {"packages": args.packages, "layouts": args.layouts, "seed": args.seed,
                 "mesh": args.mesh, "escalate": True, "sizes_mm": list(D.SIZES_MM),
                 "cooling_h": D.COOLING_H, "part_sides_mm": list(D.PART_SIDES_MM),
                 "split": D.SPLIT, "per_package_seed": "numpy default_rng([seed, package])"}
    out = D.write(args.name, args.version, result, generator)
    ds = D.load(args.name, args.version)
    c = ds.manifest["counts"]
    registry.log_run("dataset", generator, {"counts": c, "throughput": ds.manifest["throughput"]},
                     dataset={"id": ds.id, "sha256": ds.manifest["labels_sha256"]})
    print(f"\n{ds.id}: {c['labels']} labels ({c['clean']} clean, {c['escalated']} re-solved at "
          f"fine) over {c['packages']} packages in {result['wall_s']:.0f} s")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
