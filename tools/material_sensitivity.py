"""
tools/material_sensitivity.py
RISC3D-AI :: how much of a reported temperature is the material data?

The mesh study (tools/mesh_convergence.py) answers "how much of this number
is the discretization?". This one answers the other half of the error
budget: the conductivities are literature values with their own spread, so
what does the peak do when they move?

Each property is perturbed on its own, up and down, and the peak is re-solved
with everything else held fixed:

    silicon conductivity     +/- 10 %   (temperature dependence, doping, and
                                         the spread between published values)
    bump/underfill composite +/- 30 %   (the least certain layer: bump pitch,
                                         underfill filler loading, voids)

The band reported is the largest one-sided shift, and the combined budget is
the root-sum-square of the material band and the mesh band -- independent
sources, so they add in quadrature rather than linearly.

    python tools/material_sensitivity.py       # writes validation/

Verification, not validation: this says how sensitive the model is to its
inputs, not whether those inputs describe a real package.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config_loader import (  # noqa: E402
    BoundaryConditionsSection, CaseSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection,
)
from src.experiment import (  # noqa: E402
    ChipletDef, Placement, build_stack, package_from_config, solve_placement,
)
from src.thermal_solver import ThermalSolver  # noqa: E402
from tools.mesh_convergence import CASES, REFERENCE_PARTS, SIZE_MM, STACK_UM  # noqa: E402

for name in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder", "RISC3D-AI.ConfigLoader"):
    logging.getLogger(name).setLevel(logging.WARNING)

OUT_DIR = ROOT / "validation"
CELL_MM = 0.5
NZ = 12
PERTURBATIONS = {
    "silicon_k": 0.10,
    "composite_k": 0.30,
}


def peak_with(scale_silicon: float, scale_composite: float, h_w_m2k: float,
              ambient_c: float, power_scale: float) -> float:
    """Peak temperature with the two conductivity families scaled.

    The scaling is applied to the assembled maps, so it perturbs exactly what
    the solver uses and nothing else: same mesh, same power, same boundary.
    """
    n = int(round(SIZE_MM / CELL_MM))
    config = PipelineConfig(
        grid=GridSection(nx=n, ny=n, nz=NZ, dx=CELL_MM * 1e-3, dy=CELL_MM * 1e-3,
                         dz=sum(STACK_UM) * 1e-6 / NZ),
        boundary_conditions=BoundaryConditionsSection(
            h_convective=h_w_m2k, t_ambient_c=ambient_c,
            bottom_bc="neumann", top_bc="robin", side_bc="neumann"),
        stack=StackSection(interposer_um=STACK_UM[0], bump_um=STACK_UM[1], die_um=STACK_UM[2]),
        power=PowerSection(die_power_w=0.0, die_footprint_frac=0.5),
        solver=SolverSection(), visualization=VisualizationSection(),
        case=CaseSection(name="material-sensitivity"),
    )
    config.validate()
    package = package_from_config(config)
    grid_builder = build_stack(config)
    grid_builder.validate_grid()
    arrays = grid_builder.export_arrays()

    # The composite layer is the one whose in-plane and through-plane
    # conductivities differ; silicon is isotropic. Tell them apart by that,
    # rather than by index, so the split survives a stack change.
    kx, ky, kz = arrays["kx_map"], arrays["ky_map"], arrays["kz_map"]
    composite = ~np.isclose(kx, kz, rtol=1e-6)
    for array in (kx, ky, kz):
        array[composite] *= scale_composite
        array[~composite] *= scale_silicon

    solver = ThermalSolver.from_grid_builder(grid_builder)
    catalog = tuple(ChipletDef(id=p.id, name=p.id.upper(), width_um=p.w_mm * 1e3,
                               height_um=p.h_mm * 1e3, power_w=p.power_w * power_scale)
                    for p in REFERENCE_PARTS)
    placement = Placement(pitch_m=CELL_MM * 1e-3,
                          origins_m=tuple((p.id, p.x_mm * 1e-3, p.y_mm * 1e-3)
                                          for p in REFERENCE_PARTS))
    return float(solve_placement(solver, package, catalog, placement, "sensitivity").peak_c)


def mesh_band_k(case_name: str) -> float:
    try:
        study = json.loads((OUT_DIR / "mesh_convergence.json").read_text(encoding="utf-8"))
        return float(study["cases"][case_name]["gci_fine_k"])
    except (OSError, ValueError, KeyError):
        return 0.0


def main() -> None:
    report: Dict = {
        "what": "one-at-a-time conductivity perturbations on the reference package",
        "cell_mm": CELL_MM,
        "perturbations": PERTURBATIONS,
        "cases": {},
    }
    for case_name, case in CASES.items():
        base = peak_with(1.0, 1.0, case["h_w_m2k"], case["ambient_c"], case["scale"])
        entry: Dict = {"baseline_peak_c": base, "shifts_k": {}}
        for prop, delta in PERTURBATIONS.items():
            shifts: List[float] = []
            for sign in (+1.0, -1.0):
                factor = 1.0 + sign * delta
                silicon = factor if prop == "silicon_k" else 1.0
                composite = factor if prop == "composite_k" else 1.0
                peak = peak_with(silicon, composite, case["h_w_m2k"], case["ambient_c"], case["scale"])
                shifts.append(peak - base)
                print(f"  {case_name:>10} {prop:>13} {sign * delta:+.0%}  "
                      f"peak {peak:8.4f} degC  ({peak - base:+.3f} K)")
            entry["shifts_k"][prop] = {"up": shifts[0], "down": shifts[1],
                                       "band_k": max(abs(shifts[0]), abs(shifts[1]))}
        material_band = math.sqrt(sum(v["band_k"] ** 2 for v in entry["shifts_k"].values()))
        mesh = mesh_band_k(case_name)
        entry["material_band_k"] = material_band
        entry["mesh_band_k"] = mesh
        entry["combined_band_k"] = math.hypot(material_band, mesh)
        entry["rise_k"] = base - case["ambient_c"]
        entry["combined_band_pct_of_rise"] = (100.0 * entry["combined_band_k"] / entry["rise_k"]
                                              if entry["rise_k"] else 0.0)
        report["cases"][case_name] = entry

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / "material_sensitivity.json"
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {target}")
    for name, entry in report["cases"].items():
        print(f"  {name}: materials ±{entry['material_band_k']:.2f} K, mesh ±{entry['mesh_band_k']:.2f} K, "
              f"combined ±{entry['combined_band_k']:.2f} K "
              f"({entry['combined_band_pct_of_rise']:.1f} % of the rise)")


if __name__ == "__main__":
    main()
