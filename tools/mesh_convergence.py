"""
tools/mesh_convergence.py
RISC3D-AI :: how much of a reported temperature is the mesh?

Runs one reference package at three lateral mesh densities (2, 1 and 0.5 mm
cells), fits the observed order of convergence, and reports the Grid
Convergence Index -- the standard error band for a discretized solution
(Roache 1994; ASME V&V 20). The result is what lets the product print

    peak 70.23 degC  +/- 0.05 K (mesh)

instead of an unqualified number.

It also measures a SEPARATE sensitivity that is not discretization error:
the device layer where power is deposited is one cell thick, so refining the
vertical mesh thins the source. That is a modelling idealization (a real
device layer is under a micrometre), so it is reported on its own rather
than folded into the mesh band.

    python tools/mesh_convergence.py            # writes validation/
    python tools/mesh_convergence.py --quick    # two meshes, no GCI

Both numbers are verification, not validation: they say how well this model
is solved, not how well it describes real hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from src.config_loader import (  # noqa: E402
    BoundaryConditionsSection, CaseSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection,
)
from src.experiment import (  # noqa: E402
    ChipletDef, Placement, build_stack, package_from_config, solve_placement,
    validate_placement,
)
from src.thermal_solver import ThermalSolver  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT_DIR = ROOT / "validation"
GCI_SAFETY_FACTOR = 1.25          # Roache's Fs for three-mesh studies
SIZE_MM = 20.0
STACK_UM = (100.0, 40.0, 100.0)


@dataclass(frozen=True)
class Part:
    id: str
    w_mm: float
    h_mm: float
    power_w: float
    x_mm: float
    y_mm: float


# Geometry on a 2 mm lattice, so the SAME physical package is representable
# exactly at every mesh density: no chiplet edge ever lands mid-cell.
REFERENCE_PARTS = (
    Part("gpu", 6.0, 4.0, 4.67, 12.0, 2.0),
    Part("cpu", 4.0, 4.0, 3.33, 6.0, 12.0),
    Part("hbm", 10.0, 10.0, 1.33, 0.0, 0.0),
    Part("npu", 4.0, 4.0, 2.67, 0.0, 12.0),
)

CASES = {
    "passive": dict(h_w_m2k=1000.0, ambient_c=25.0, scale=1.0),
    "coldplate": dict(h_w_m2k=25000.0, ambient_c=25.0, scale=12.5),
}


def solve_once(cell_mm: float, nz: int, h_w_m2k: float, ambient_c: float,
               scale: float) -> Tuple[float, int, float]:
    """Peak temperature (degC) for the reference package on this mesh."""
    n = int(round(SIZE_MM / cell_mm))
    dz = sum(STACK_UM) * 1e-6 / nz
    config = PipelineConfig(
        grid=GridSection(nx=n, ny=n, nz=nz, dx=cell_mm * 1e-3, dy=cell_mm * 1e-3, dz=dz),
        boundary_conditions=BoundaryConditionsSection(
            h_convective=h_w_m2k, t_ambient_c=ambient_c,
            bottom_bc="neumann", top_bc="robin", side_bc="neumann"),
        stack=StackSection(interposer_um=STACK_UM[0], bump_um=STACK_UM[1], die_um=STACK_UM[2]),
        power=PowerSection(die_power_w=0.0, die_footprint_frac=0.5),
        solver=SolverSection(), visualization=VisualizationSection(),
        case=CaseSection(name="mesh-convergence"),
    )
    config.validate()
    package = package_from_config(config)
    grid_builder = build_stack(config)
    grid_builder.validate_grid()
    solver = ThermalSolver.from_grid_builder(grid_builder)
    catalog = tuple(ChipletDef(id=p.id, name=p.id.upper(), width_um=p.w_mm * 1e3,
                               height_um=p.h_mm * 1e3, power_w=p.power_w * scale)
                    for p in REFERENCE_PARTS)
    placement = Placement(pitch_m=cell_mm * 1e-3,
                          origins_m=tuple((p.id, p.x_mm * 1e-3, p.y_mm * 1e-3)
                                          for p in REFERENCE_PARTS))
    violations = validate_placement(package, catalog, placement)
    if violations:
        raise SystemExit(f"reference placement is invalid at {cell_mm} mm: {violations}")
    t0 = time.perf_counter()
    result = solve_placement(solver, package, catalog, placement, f"{cell_mm} mm")
    return float(result.peak_c), package.nx * package.ny * package.nz, time.perf_counter() - t0


def gci(fine: float, medium: float, coarse: float, ratio: float, ambient_c: float) -> Dict:
    """Observed order and Grid Convergence Index on the finest mesh."""
    d21 = medium - fine
    d32 = coarse - medium
    if abs(d21) < 1e-12:
        return {"order_p": None, "gci_fine_pct": 0.0, "gci_fine_k": 0.0,
                "extrapolated_c": fine, "note": "the two finest meshes agree to 1e-12 K"}
    p = math.log(abs(d32 / d21)) / math.log(ratio)
    extrapolated = fine - d21 / (ratio ** p - 1.0)      # Richardson
    rise = fine - ambient_c
    relative = abs(d21 / rise) if rise else 0.0
    band = GCI_SAFETY_FACTOR * relative / (ratio ** p - 1.0)
    return {
        "order_p": p,
        "gci_fine_pct": 100.0 * band,
        "gci_fine_k": band * rise,
        "extrapolated_c": extrapolated,
    }


def run(quick: bool) -> Dict:
    meshes = [2.0, 1.0] if quick else [2.0, 1.0, 0.5]
    report: Dict[str, Dict] = {
        "what": "lateral mesh refinement on a fixed reference package",
        "reference_package": {
            "size_mm": SIZE_MM, "stack_um": list(STACK_UM), "nz": 12,
            "chiplets": [p.__dict__ for p in REFERENCE_PARTS],
        },
        "safety_factor": GCI_SAFETY_FACTOR,
        "cases": {},
    }
    for case_name, case in CASES.items():
        peaks: List[Tuple[float, float, int, float]] = []
        for cell in meshes:
            peak, cells, seconds = solve_once(cell, 12, case["h_w_m2k"], case["ambient_c"], case["scale"])
            peaks.append((cell, peak, cells, seconds))
            print(f"  {case_name:>10} {cell:>4} mm  {cells:>7,} cells  peak {peak:8.4f} degC  {seconds:5.2f} s")
        entry: Dict = {
            "h_w_m2k": case["h_w_m2k"],
            "ambient_c": case["ambient_c"],
            "total_power_w": sum(p.power_w for p in REFERENCE_PARTS) * case["scale"],
            "meshes": [{"cell_mm": c, "peak_c": pk, "cells": n, "solve_s": s} for c, pk, n, s in peaks],
        }
        if len(peaks) == 3:
            coarse, medium, fine = peaks[0][1], peaks[1][1], peaks[2][1]
            entry.update(gci(fine, medium, coarse, 2.0, case["ambient_c"]))
            # The band a COARSER mesh carries, for the packages the app cannot
            # mesh at 0.5 mm: scale the same relative difference.
            rise = fine - case["ambient_c"]
            entry["band_by_cell_mm"] = {
                "0.5": entry["gci_fine_k"],
                "1": GCI_SAFETY_FACTOR * abs(medium - fine),
                "2": GCI_SAFETY_FACTOR * abs(coarse - fine),
            }
            entry["band_pct_of_rise_by_cell_mm"] = {
                k: (100.0 * v / rise if rise else 0.0) for k, v in entry["band_by_cell_mm"].items()
            }
        report["cases"][case_name] = entry

    # Separate: the device layer is one cell thick, so a finer z-mesh thins it.
    print("  device-layer sensitivity (nz 12 -> 24, source 20 um -> 10 um):")
    device: Dict[str, Dict] = {}
    for case_name, case in CASES.items():
        coarse_peak, _, _ = solve_once(1.0, 12, case["h_w_m2k"], case["ambient_c"], case["scale"])
        fine_peak, _, _ = solve_once(1.0, 24, case["h_w_m2k"], case["ambient_c"], case["scale"])
        device[case_name] = {
            "peak_nz12_c": coarse_peak, "peak_nz24_c": fine_peak,
            "delta_k": fine_peak - coarse_peak,
        }
        print(f"  {case_name:>10} {coarse_peak:8.4f} -> {fine_peak:8.4f} degC "
              f"({fine_peak - coarse_peak:+.4f} K)")
    report["device_layer_sensitivity"] = {
        "what": "power is deposited in ONE cell at the bumped face, so halving dz "
                "halves the source thickness. This is a modelling idealization "
                "(a real device layer is under a micrometre), not discretization "
                "error, and is reported separately for that reason.",
        "cases": device,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="two meshes, no GCI")
    args = parser.parse_args()

    print("Mesh convergence on the reference package (2 / 1 / 0.5 mm cells):")
    report = run(args.quick)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "mesh_convergence.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}")
    for name, case in report["cases"].items():
        if "order_p" in case and case["order_p"]:
            print(f"  {name}: observed order {case['order_p']:.2f}, "
                  f"GCI at 0.5 mm = {case['gci_fine_k']:.3f} K "
                  f"({case['gci_fine_pct']:.2f} % of the rise)")


if __name__ == "__main__":
    main()
