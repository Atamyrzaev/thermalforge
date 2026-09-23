"""
ai/dataset.py
RISC3D-AI :: versioned label sets, made by the verified solver.

A claim about a model's accuracy is only as good as the data it was measured
on, so a dataset here is an artifact with an identity, not a folder of files:

    data/datasets/<name>/<version>/
        manifest.json      schema, generator settings and seed, commit, counts,
                           the split by PACKAGE, and the sha256 of the labels
        labels.jsonl.gz    one verified label per line

Every label is one solve that passed its residual, energy-balance and
convective-floor checks, with its band and its checks written beside it. A
label whose peak the mesh cannot resolve (a hotspot finer than a cell) is
re-solved at the fine mesh; one still unresolved there is kept, marked
clean=False, and never trained on -- a surrogate trained on it would learn the
mesh's error. load() refuses a dataset whose labels no longer hash to its
manifest, so a model card that names a dataset names exactly those labels.

The split is by package, never by layout: the S4 gate is written for held-out
PACKAGES, and a model that has seen a package has seen most of its answer.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ai.records import git_commit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "datasets"
SCHEMA_VERSION = 1

SIZES_MM = (10.0, 20.0, 30.0, 40.0)
# cooling stack -> plausible range of the coefficient on its outer face
COOLING_H = {"direct": (500.0, 5000.0), "lid": (1000.0, 10000.0),
             "lid_heatsink": (2000.0, 30000.0), "coldplate": (10000.0, 50000.0)}
PART_SIDES_MM = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
SPLIT = {"train": 0.6, "calib": 0.2, "test": 0.2}


class DatasetError(Exception):
    pass


def random_package(rng) -> dict:
    import studio.engine as E
    cooling = str(rng.choice(sorted(COOLING_H)))
    lo, hi = COOLING_H[cooling]
    raw = {"size_mm": float(rng.choice(SIZES_MM)),
           "h_w_m2k": float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), 0)),
           "ambient_c": float(np.round(rng.uniform(25.0, 45.0), 1)),
           "pitch_mm": 1.0, "cooling": cooling}
    # the stack and the per-design options vary too, or the surrogate never
    # sees a package that is not the default one
    if rng.random() < 0.35:
        raw["stack_um"] = [float(rng.choice([60, 100, 200])), float(rng.choice([20, 40, 60])),
                           float(rng.choice([60, 100, 200]))]
    if rng.random() < 0.2:
        raw["interposer_material"] = "tsv_interposer"
    if rng.random() < 0.2:
        raw["beol_um"] = float(rng.choice([5.0, 10.0]))
        stack = raw.get("stack_um", list(E.STACK_UM))
        if raw["beol_um"] >= stack[1]:
            raw.pop("beol_um")
    return raw


def random_parts(rng, size_mm: float, max_parts: int = 8) -> List[dict]:
    """1..max_parts distinct parts, legal (no overlap, 1 mm clear), on whole mm."""
    import studio.engine as E
    patterns = tuple(E.POWER_PATTERNS)
    want = int(rng.integers(1, max_parts + 1))
    parts, taken, tries = [], [], 0
    while len(parts) < want and tries < 4000:
        tries += 1
        w, h = (float(rng.choice(PART_SIDES_MM)) for _ in range(2))
        if w > size_mm or h > size_mm:
            continue
        x = float(rng.integers(0, int(size_mm - w) + 1))
        y = float(rng.integers(0, int(size_mm - h) + 1))
        if any(x < ox + ow + 1 and ox < x + w + 1 and y < oy + oh + 1 and oy < y + h + 1
               for ox, oy, ow, oh in taken):
            continue
        taken.append((x, y, w, h))
        n = len(parts)
        parts.append({"id": f"p{n}", "name": f"P{n}", "w_mm": w, "h_mm": h,
                      "power_w": float(np.round(rng.uniform(0.5, 15.0), 2)),
                      "power_pattern": str(rng.choice(patterns)), "x_mm": x, "y_mm": y})
    return parts


def label_from(out: dict, package: dict, parts: List[dict], package_id: int, mesh: str,
               first_flagged: bool, escalated: bool, commit: str, ms: float) -> dict:
    u = out["metrics"]["uncertainty"] or {}
    return {
        "schema": SCHEMA_VERSION, "commit": commit, "package_id": package_id,
        "package": package, "chiplets": parts,
        "peak_c": out["peak"]["t_c"], "rise_k": out["metrics"]["rise_k"],
        "peak_xy_mm": [out["peak"]["x_mm"], out["peak"]["y_mm"]],
        "hottest_chiplet": out.get("hottest_chiplet"),
        "chiplet_peaks_c": {r["id"]: r["max_c"] for r in out["chiplets"]},
        "band_k": u.get("combined_k"),
        "band_terms_k": {k: u.get(k) for k in ("mesh_k", "material_k", "boundary_k")},
        "peak_is_lower_bound": bool(u.get("peak_is_lower_bound")),
        "flagged_at_first_mesh": first_flagged, "escalated": escalated,
        "clean": not bool(u.get("peak_is_lower_bound")),
        "checks": {"residual": out["verification"]["residual"],
                   "energy_imbalance_k": out["verification"]["energy_imbalance_k"],
                   "passed": True},
        "mesh": mesh, "cell_mm": out["grid"].get("cell_mm"), "unknowns": out["grid"]["unknowns"],
        "solve_ms": ms,
    }


def _package_labels(args) -> Dict:
    """One package and its layouts, from its OWN random stream, so the
    dataset does not depend on how packages are shared out between workers."""
    import logging
    import studio.engine as E
    for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
        logging.getLogger(name).setLevel(logging.WARNING)
    p_index, layouts, seed, mesh, escalate, commit = args
    rng = np.random.default_rng([seed, p_index])
    refused = {"package": 0, "layout": 0}
    while True:
        raw = random_package(rng)
        raw["mesh"] = mesh
        try:
            spec = E.PackageSpec.parse(raw)
            break
        except E.StudioError:
            refused["package"] += 1
    cache = E.ModelCache(capacity=2)
    t0 = time.perf_counter()
    cache.get(spec)
    build_s = time.perf_counter() - t0
    labels = []
    for _ in range(layouts):
        parts = random_parts(rng, spec.size_mm)
        try:
            chips = E.parse_chiplets(parts, spec)
        except E.StudioError:
            refused["layout"] += 1
            continue
        t0 = time.perf_counter()
        out = E.solve(cache, spec, chips)          # raises if a per-solve check fails
        if not out.get("ok"):
            refused["layout"] += 1
            continue
        used, first = mesh, bool((out["metrics"]["uncertainty"] or {}).get("peak_is_lower_bound"))
        if first and escalate and mesh != "fine":
            fine = E.PackageSpec.parse(dict(raw, mesh="fine"))
            if fine.cell_mm < spec.cell_mm:
                out = E.solve(cache, fine, E.parse_chiplets(parts, fine))
                used = "fine"
        ms = (time.perf_counter() - t0) * 1e3
        labels.append(label_from(out, raw, parts, p_index, used, first, used != mesh, commit, ms))
    return {"p_index": p_index, "raw": raw, "labels": labels, "refused": refused, "build_s": build_s}


def generate(n_packages: int, layouts: int, seed: int, mesh: str = "standard",
             escalate: bool = True, workers: int = 1,
             on_package: Optional[Callable[[int, dict, int], None]] = None) -> Dict:
    """Labels from the verified solver: many layouts per package, because a
    package costs a factorization and a layout one back-substitution.
    Packages are independent and seeded separately, so `workers` processes
    give the same dataset as one."""
    commit = git_commit()
    jobs = [(p, layouts, seed, mesh, escalate, commit) for p in range(n_packages)]
    results = []
    t_all = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            r = _package_labels(job)
            results.append(r)
            if on_package is not None:
                on_package(r["p_index"], r["raw"], len(r["labels"]))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for r in pool.map(_package_labels, jobs):
                results.append(r)
                if on_package is not None:
                    on_package(r["p_index"], r["raw"], len(r["labels"]))
    wall = time.perf_counter() - t_all
    results.sort(key=lambda r: r["p_index"])
    labels = [lab for r in results for lab in r["labels"]]
    refused = {"package": sum(r["refused"]["package"] for r in results),
               "layout": sum(r["refused"]["layout"] for r in results)}
    return {"labels": labels, "refused": refused, "wall_s": wall, "commit": commit,
            "package_build_s": [r["build_s"] for r in results], "workers": workers}


def split_packages(n_packages: int, seed: int) -> Dict[str, List[int]]:
    rng = np.random.default_rng(seed + 1)
    ids = rng.permutation(n_packages).tolist()
    n_train = int(round(SPLIT["train"] * n_packages))
    n_calib = int(round(SPLIT["calib"] * n_packages))
    return {"train": sorted(ids[:n_train]), "calib": sorted(ids[n_train:n_train + n_calib]),
            "test": sorted(ids[n_train + n_calib:])}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name: str, version: int, result: Dict, generator: Dict, root: Path = DATA) -> Path:
    out = root / name / f"v{version}"
    if out.exists():
        raise DatasetError(f"{name} v{version} already exists; versions are immutable")
    out.mkdir(parents=True)
    labels = result["labels"]
    path = out / "labels.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for lab in labels:
            f.write(json.dumps(lab) + "\n")
    n_packages = generator["packages"]
    manifest = {
        "schema": SCHEMA_VERSION, "name": name, "version": version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "commit": result["commit"],
        "generator": generator,
        "counts": {"labels": len(labels), "clean": sum(l["clean"] for l in labels),
                   "flagged_at_first_mesh": sum(l["flagged_at_first_mesh"] for l in labels),
                   "escalated": sum(l["escalated"] for l in labels),
                   "packages": n_packages, "refused": result["refused"]},
        "throughput": {"wall_s": result["wall_s"], "workers": result.get("workers", 1),
                       "labels_per_s": len(labels) / result["wall_s"] if result["wall_s"] else None,
                       "note": "as generated, with other work on the machine; the clean "
                               "benchmark is tools/s4_readiness.py"},
        "split": split_packages(n_packages, generator["seed"]),
        "labels_sha256": _sha256(path),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


@dataclass
class Dataset:
    manifest: Dict
    labels: List[Dict]

    @property
    def id(self) -> str:
        return f"{self.manifest['name']}/v{self.manifest['version']}"

    def part(self, which: str, clean_only: bool = True) -> List[Dict]:
        ids = set(self.manifest["split"][which])
        return [l for l in self.labels if l["package_id"] in ids and (l["clean"] or not clean_only)]


def load(name: str, version: Optional[int] = None, root: Path = DATA) -> Dataset:
    base = root / name
    if version is None:
        versions = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
        if not versions:
            raise DatasetError(f"no dataset named {name}")
        version = versions[-1]
    d = base / f"v{version}"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    path = d / "labels.jsonl.gz"
    if _sha256(path) != manifest["labels_sha256"]:
        raise DatasetError(f"{name} v{version}: labels do not match the manifest's sha256 -- "
                           f"refusing to use a dataset that changed after it was made")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        labels = [json.loads(line) for line in f if line.strip()]
    return Dataset(manifest=manifest, labels=labels)
