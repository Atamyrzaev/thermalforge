"""
studio/scenarios.py
RISC3D Studio :: built-in demo scenarios.

The first one is read from configs/config.json -- the project's canonical
experiment -- so the app and the command-line pipeline describe the same
chip. The others reuse its catalog under different cooling, or add an
illustrative RISC-V many-core layout; their powers and sizes are
representative, not measurements of a specific product.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
CANONICAL = ROOT / "configs" / "config.json"

PALETTE = {"CPU": "#4f8cff", "GPU": "#35c48b", "HBM": "#b27cff", "NPU": "#ffb347",
           "RV": "#5ad1e6", "SRAM": "#c38cff", "SERDES": "#ff7aa8", "PMIC": "#9fb4c7"}


def _colour(name: str) -> str:
    upper = name.upper()
    for key, colour in PALETTE.items():
        if upper.startswith(key):
            return colour
    return "#8aa0b8"


# What the compute actually looks like inside each part. A chiplet is not a
# uniform heater, and assuming it is under-reports the junction: on the NPU
# below, a MAC array carrying 70 % of the power in 40 % of the area runs 13 K
# hotter than the same watts spread evenly.
PATTERNS = {"NPU": "mac_array", "GPU": "two_engines", "CPU": "quad_cores"}


def _pattern(name: str) -> str:
    upper = name.upper()
    for key, pattern in PATTERNS.items():
        if upper.startswith(key):
            return pattern
    return "uniform"


def _canonical_chiplets() -> List[dict]:
    raw = json.loads(CANONICAL.read_text(encoding="utf-8"))
    origins = raw["initial_placement"]["origins_um"]
    out = []
    for c in raw["chiplets"]:
        name = c["name"].replace("_", " ")
        out.append({"id": c["id"], "name": name,
                    "w_mm": c["width_um"] / 1e3, "h_mm": c["height_um"] / 1e3,
                    "power_w": c["power_w"], "x_mm": origins[c["id"]][0] / 1e3,
                    "y_mm": origins[c["id"]][1] / 1e3, "color": _colour(c["name"]),
                    "power_pattern": _pattern(name)})
    return out


def scenarios() -> List[dict]:
    raw = json.loads(CANONICAL.read_text(encoding="utf-8"))
    bcs = raw["boundary_conditions"]
    size_mm = raw["grid"]["nx"] * raw["grid"]["dx"] * 1e3
    base = _canonical_chiplets()
    total = sum(c["power_w"] for c in base)

    def scaled(target_w: float) -> List[dict]:
        k = target_w / total
        return [{**c, "power_w": c["power_w"] * k} for c in base]

    rv_cores = [{"id": f"rv{i}", "name": f"RV core {i}", "w_mm": 3.0, "h_mm": 3.0,
                 "power_w": 2.5, "x_mm": x, "y_mm": y, "color": _colour("RV")}
                for i, (x, y) in enumerate([(0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)])]
    return [
        {
            "id": "edge-ai-soc",
            "title": "Edge AI SoC - passive cooling",
            "summary": "The project's canonical experiment: CPU, GPU, HBM and NPU chiplets "
                       "(12 W) on a 20 x 20 mm interposer with a passive heat spreader.",
            "package": {"size_mm": size_mm, "h_w_m2k": bcs["h_convective"],
                        "ambient_c": bcs["t_ambient_c"],
                        "pitch_mm": raw["initial_placement"]["pitch_um"] / 1e3},
            "chiplets": base,
        },
        {
            "id": "datacenter-coldplate",
            "title": "Datacenter accelerator - liquid cold plate",
            "summary": "The same four chiplets at 150 W under a liquid cold plate. Each "
                       "chiplet's own power density dominates here, so placement gains "
                       "are small: the optimizer proves how little there is to win.",
            "package": {"size_mm": size_mm, "h_w_m2k": 25000.0, "ambient_c": 25.0,
                        "pitch_mm": 2.0},
            "chiplets": scaled(150.0),
        },
        {
            "id": "riscv-manycore",
            "title": "RISC-V many-core - forced air",
            "summary": "Illustrative: four identical RISC-V cores, an NPU and SRAM packed "
                       "into one corner under forced-air cooling.",
            "package": {"size_mm": 20.0, "h_w_m2k": 2500.0, "ambient_c": 35.0,
                        "pitch_mm": 2.0},
            "chiplets": rv_cores + [
                {"id": "npu", "name": "NPU", "w_mm": 5.0, "h_mm": 5.0, "power_w": 8.0,
                 "x_mm": 8.0, "y_mm": 0.0, "color": _colour("NPU"),
                 "power_pattern": "mac_array"},
                {"id": "sram", "name": "SRAM", "w_mm": 6.0, "h_mm": 6.0, "power_w": 2.0,
                 "x_mm": 0.0, "y_mm": 8.0, "color": _colour("SRAM")},
            ],
        },
    ]
