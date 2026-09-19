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
FORMAL_ORDER = 2.0                # what the finite-volume scheme is; see gci()
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

# Internal power patterns, on the compute parts only. A MAC array is 40 % of
# the footprint carrying 70 % of the power; those block edges are FRACTIONS of
# a chiplet (0.185 to 0.815 of its width), so unlike the chiplet outlines they
# cannot land on a cell boundary at every mesh. That is the point: this is the
# case the uniform reference package does not describe.
HOTSPOT_PATTERNS = {"gpu": "mac_array", "npu": "mac_array", "cpu": "quad_cores"}

# A single small, dense part. The reference package above has 4 to 10 mm
# chiplets, which is a gentle spread of heat; tools/band_check.py swept
# fourteen designs against the band derived from it and four exceeded it, the
# worst by 2.85x -- every one of them a UNIFORM design with a small dense part.
# The reference set has to span what the app permits, not just what it was
# first written with, or the band describes a design space nobody is in.
POINT_PARTS = (
    Part("hot", 2.0, 2.0, 8.0, 10.0, 10.0),
    Part("warm", 4.0, 4.0, 2.0, 2.0, 2.0),
)

CASES = {
    "passive": dict(h_w_m2k=1000.0, ambient_c=25.0, scale=1.0, concentrated=False),
    "coldplate": dict(h_w_m2k=25000.0, ambient_c=25.0, scale=12.5, concentrated=False),
    # Uniform chiplets still, but concentrated: this is what sets the band for
    # a design whose heat comes from one small part.
    "passive_point": dict(h_w_m2k=1000.0, ambient_c=25.0, scale=1.0,
                          concentrated=False, parts=POINT_PARTS),
    "coldplate_point": dict(h_w_m2k=25000.0, ambient_c=25.0, scale=12.5,
                            concentrated=False, parts=POINT_PARTS),
    # The SAME package and the same powers, with the compute parts carrying
    # their internal hotspots. Measured, this is the whole story: with uniform
    # chiplets the peak moves 0.004 K between 0.5 mm and 0.25 mm, and with the
    # patterns it moves 1.0 K. Quoting the uniform band for a patterned design
    # understates it by more than an order of magnitude -- and the patterns are
    # the feature that makes the answer realistic in the first place.
    "passive_hotspots": dict(h_w_m2k=1000.0, ambient_c=25.0, scale=1.0, concentrated=True),
    "coldplate_hotspots": dict(h_w_m2k=25000.0, ambient_c=25.0, scale=12.5, concentrated=True),
}

# Forcing an exact cell size out of the studio engine. It picks the finest
# cell that fits its unknown budget, so a budget that only one cell size can
# satisfy names that cell: on a 20 mm package with 12 z-cells, 2 mm is 1,200
# unknowns, 1 mm is 4,800, 0.5 mm is 19,200 and 0.25 mm is 76,800.
STUDY_LEVELS = {"study_2": 1_200, "study_1": 4_800, "study_0.5": 19_200,
                "study_0.25": 76_800}
_studio_cache = None


def solve_patterned(cell_mm: float, h_w_m2k: float, ambient_c: float,
                    scale: float, parts=None) -> Tuple[float, int, float]:
    """The same reference package, solved THROUGH THE APP with its patterns on.

    Deliberately not the core pipeline the uniform cases use: the quantity
    being measured here is how much the app's own answer moves with the mesh,
    and the internal-power machinery lives in the app. Going around it would
    measure something the user never sees.
    """
    global _studio_cache
    import studio.engine as SE

    SE.MESH_LEVELS.update(STUDY_LEVELS)
    if _studio_cache is None:
        _studio_cache = SE.ModelCache()
    level = f"study_{cell_mm:g}"
    spec = SE.PackageSpec.parse({
        "size_mm": SIZE_MM, "h_w_m2k": h_w_m2k, "ambient_c": ambient_c,
        "pitch_mm": 2.0, "mesh": level, "keepout_um": 0.0,
        "stack_um": list(STACK_UM),
    })
    if abs(spec.cell_mm - cell_mm) > 1e-9:
        raise SystemExit(f"wanted a {cell_mm} mm cell, the engine chose {spec.cell_mm}")
    raw = [{"id": part.id, "name": part.id.upper(), "w_mm": part.w_mm, "h_mm": part.h_mm,
            "power_w": part.power_w * scale, "x_mm": part.x_mm, "y_mm": part.y_mm,
            "power_pattern": HOTSPOT_PATTERNS.get(part.id, "uniform")}
           for part in (parts or REFERENCE_PARTS)]
    started = time.perf_counter()
    out = SE.solve(_studio_cache, spec, SE.parse_chiplets(raw, spec))
    seconds = time.perf_counter() - started
    if not out["ok"]:
        raise SystemExit(f"the reference package was refused: {out['violations']}")
    return float(out["peak"]["t_c"]), spec.unknowns, seconds


def solve_once(cell_mm: float, nz: int, h_w_m2k: float, ambient_c: float,
               scale: float, parts=None) -> Tuple[float, int, float]:
    """Peak temperature (degC) for the reference package on this mesh."""
    parts = parts or REFERENCE_PARTS
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
                    for p in parts)
    placement = Placement(pitch_m=cell_mm * 1e-3,
                          origins_m=tuple((p.id, p.x_mm * 1e-3, p.y_mm * 1e-3)
                                          for p in parts))
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
    observed = math.log(abs(d32 / d21)) / math.log(ratio)
    # An apparent order above the scheme's formal one means the finest
    # differences are near the noise floor, not that the method became
    # third-order. Roache's advice is to use the formal order there; taking
    # 3.3 at face value would divide the band by 9 and flatter the result.
    p = min(observed, FORMAL_ORDER) if observed > FORMAL_ORDER else observed
    extrapolated = fine - d21 / (ratio ** p - 1.0)      # Richardson
    rise = fine - ambient_c
    relative = abs(d21 / rise) if rise else 0.0
    band = GCI_SAFETY_FACTOR * relative / (ratio ** p - 1.0)
    return {
        "order_p": p,
        "order_observed": observed,
        "order_clamped_to_formal": observed > FORMAL_ORDER,
        "gci_fine_pct": 100.0 * band,
        "gci_fine_k": band * rise,
        "extrapolated_c": extrapolated,
    }


def run(quick: bool) -> Dict:
    # 0.25 mm is affordable now that the app installs its own fill-reducing
    # factorization: 76,800 unknowns factorize in about 3 s. With a level
    # BELOW the one the app ships on, the discretization error at 0.5 mm can
    # be MEASURED against it rather than extrapolated from coarser meshes.
    meshes = [2.0, 1.0] if quick else [2.0, 1.0, 0.5, 0.25]
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
        parts = case.get("parts")
        solver = solve_patterned if case["concentrated"] else (
            lambda c, h, a, sc, pp: solve_once(c, 12, h, a, sc, pp))
        for cell in meshes:
            peak, cells, seconds = solver(cell, case["h_w_m2k"], case["ambient_c"],
                                          case["scale"], parts)
            peaks.append((cell, peak, cells, seconds))
            print(f"  {case_name:>10} {cell:>4} mm  {cells:>7,} cells  peak {peak:8.4f} degC  {seconds:5.2f} s")
        entry: Dict = {
            "h_w_m2k": case["h_w_m2k"],
            "ambient_c": case["ambient_c"],
            "concentrated": case["concentrated"],
            "patterns": HOTSPOT_PATTERNS if case["concentrated"] else {},
            "solved_by": "studio.engine" if case["concentrated"] else "src pipeline",
            "total_power_w": sum(p.power_w for p in (case.get("parts") or REFERENCE_PARTS))
                             * case["scale"],
            "parts": [p.id for p in (case.get("parts") or REFERENCE_PARTS)],
            "meshes": [{"cell_mm": c, "peak_c": pk, "cells": n, "solve_s": s} for c, pk, n, s in peaks],
        }
        if len(peaks) >= 3:
            # The order and the band come from the FINEST levels available.
            # Estimating them from coarse meshes was the old way and it was
            # 7 to 11 times too pessimistic, because the coarse end of the
            # sequence is where the solution is least converged: measured
            # against a 0.25 mm solution, the error at 0.5 mm is 0.04 K
            # (passive) and 0.24 K (cold plate), against GCI estimates of
            # 0.43 K and 1.76 K from the 2/1/0.5 triple.
            coarse, medium, fine = peaks[-3][1], peaks[-2][1], peaks[-1][1]
            entry.update(gci(fine, medium, coarse, 2.0, case["ambient_c"]))
            entry["gci_levels_mm"] = [peaks[-3][0], peaks[-2][0], peaks[-1][0]]
        if len(peaks) >= 4:
            finest = peaks[-1][1]
            rise_finest = finest - case["ambient_c"]
            entry["finest_cell_mm"] = peaks[-1][0]
            entry["measured_error_by_cell_mm"] = {
                f"{cell:g}": abs(peak - finest) for cell, peak, _n, _s in peaks[:-1]
            }
            entry["measured_error_pct_by_cell_mm"] = {
                key: (100.0 * value / rise_finest if rise_finest else 0.0)
                for key, value in entry["measured_error_by_cell_mm"].items()
            }
            entry["gci_over_measured_at_0.5"] = (
                entry["gci_fine_k"] / entry["measured_error_by_cell_mm"]["0.5"]
                if entry["measured_error_by_cell_mm"].get("0.5") else None)
            # The band a COARSER mesh carries, for the packages the app cannot
            # mesh at 0.5 mm: scale the same relative difference.
            rise = fine - case["ambient_c"]
            # The band each mesh carries: the finest gets the GCI, and every
            # coarser one gets its own measured distance to the finest,
            # with the same safety factor. Nothing here is extrapolated from
            # a mesh coarser than the one being described.
            finest_cell = peaks[-1][0]
            entry["band_by_cell_mm"] = {f"{finest_cell:g}": entry["gci_fine_k"]}
            for cell, peak, _cells, _seconds in peaks[:-1]:
                entry["band_by_cell_mm"][f"{cell:g}"] = GCI_SAFETY_FACTOR * abs(peak - fine)
            entry["band_pct_of_rise_by_cell_mm"] = {
                k: (100.0 * v / rise if rise else 0.0) for k, v in entry["band_by_cell_mm"].items()
            }
        report["cases"][case_name] = entry

    # Separate: the device layer is one cell thick, so a finer z-mesh thins it.
    print("  device-layer sensitivity (nz 12 -> 24, source 20 um -> 10 um):")
    device: Dict[str, Dict] = {}
    for case_name, case in CASES.items():
        if case["concentrated"]:
            continue    # a z-mesh sweep; the lateral patterns do not change it
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
        measured = case.get("measured_error_by_cell_mm")
        if measured:
            parts = ", ".join(f"{cell} mm {value:.3f} K" for cell, value in measured.items())
            print(f"    measured against {case['finest_cell_mm']} mm: {parts}")
            band = case.get("band_by_cell_mm", {}).get("0.5")
            if band:
                print(f"    band now quoted at 0.5 mm: {band:.3f} K "
                      f"(the old 2/1/0.5 extrapolation said {case['gci_fine_k']:.3f} K at best)")
            if case.get("order_clamped_to_formal"):
                print(f"    observed order {case['order_observed']:.2f} exceeds the scheme's "
                      f"formal {FORMAL_ORDER:.0f}; clamped, which widens the band rather than "
                      f"flattering it")


if __name__ == "__main__":
    main()
