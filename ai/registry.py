"""
ai/registry.py
ThermalForge :: model registry and experiment log.

A model is registered with a card that says what it was trained on and how it
was measured, and a version that never changes afterwards:

    data/models/<name>/v<N>/model.json     parameters (plain JSON, no pickles)
    data/models/<name>/v<N>/card.json      dataset id + sha256, training config,
                                           held-out metrics, commit, sha256 of
                                           model.json
    data/experiments.jsonl                 one line per training or evaluation
                                           run, kept or not

load() refuses a model whose parameters no longer hash to its card, and a
card that names a dataset names it by id AND by the labels' sha256, so
"measured on held-out packages" can be checked rather than believed. Runs that
did not produce a registered model are logged too: an experiment log that
only keeps the winners is a survivorship bias with a file extension.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ai.records import git_commit

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"
EXPERIMENTS = ROOT / "data" / "experiments.jsonl"


class RegistryError(Exception):
    pass


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log_run(kind: str, config: Dict[str, Any], metrics: Dict[str, Any],
            dataset: Optional[Dict[str, str]] = None, model: Optional[str] = None,
            path: Path = EXPERIMENTS) -> str:
    run_id = uuid.uuid4().hex[:12]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"run": run_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "commit": git_commit(), "kind": kind, "config": config,
                            "dataset": dataset, "model": model, "metrics": metrics},
                           default=float) + "\n")
    return run_id


def runs(path: Path = EXPERIMENTS):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def register(name: str, params: Dict[str, Any], card: Dict[str, Any],
             root: Path = MODELS) -> Tuple[int, Path]:
    base = root / name
    base.mkdir(parents=True, exist_ok=True)
    versions = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
    version = (versions[-1] + 1) if versions else 1
    out = base / f"v{version}"
    out.mkdir()
    model_text = json.dumps(params, indent=None, separators=(",", ":"), default=float)
    (out / "model.json").write_text(model_text, encoding="utf-8")
    full_card = {"name": name, "version": version, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "commit": git_commit(), "model_sha256": _digest(model_text), **card}
    (out / "card.json").write_text(json.dumps(full_card, indent=2, default=float), encoding="utf-8")
    return version, out


def load(name: str, version: Optional[int] = None,
         root: Path = MODELS) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = root / name
    versions = sorted(int(p.name[1:]) for p in base.glob("v*") if p.name[1:].isdigit())
    if not versions:
        raise RegistryError(f"no registered model named {name}")
    v = versions[-1] if version is None else version
    d = base / f"v{v}"
    if not d.exists():
        raise RegistryError(f"{name} has no version {v}")
    card = json.loads((d / "card.json").read_text(encoding="utf-8"))
    text = (d / "model.json").read_text(encoding="utf-8")
    if _digest(text) != card["model_sha256"]:
        raise RegistryError(f"{name} v{v}: parameters do not match the card's sha256")
    return json.loads(text), card
