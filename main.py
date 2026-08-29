#!/usr/bin/env python3
"""
main.py
RISC3D-AI :: End-to-End 3D IC Thermal Simulation Pipeline (Interactive CLI)

Entry point that:
  1. Prompts the user directly in the terminal for all simulation parameters
     (grid size, spacing, ambient temp, convective coefficient, die power,
     step count, safety factor) -- NO configs/config.json dependency for
     the default interactive workflow. Press Enter on any prompt to accept
     the bracketed default.
  2. Configures a multi-layer TSMC-style advanced-packaging chiplet stack
     (Interposer -> Microbump+Underfill composite -> Silicon Die with an
     active compute-block power density Q) via src.grid_builder.GridBuilder.
  3. Constructs src.thermal_solver.ThermalSolver, computes the automated
     CFL-stable timestep, and runs the explicit solver for exactly the
     requested number of steps.
  4. Renders the result as a single interactive WebGL 3D HTML file
     (orbit/zoom/pan, Z-layer slicing, click-to-inspect voxel temperatures)
     via src.visualization.ThermalVisualizer.render_interactive_webgl(),
     and opens it automatically in the default browser.
  5. Optionally (--optimize, or by answering yes to the interactive prompt
     offered once Stage 1 has finished) runs Stage 2: the automated chiplet
     placement optimizer in src.optimizer, which simulated-anneals over
     non-overlapping chiplet placements to minimize steady-state peak
     temperature, then writes and opens
     outputs/optimized_thermal_report.html.
  6. Reports wall-clock timing metrics for each pipeline phase.

STAGE 2 TIMING NOTE:
Stage 2 scores candidates with ThermalSolver.solve_steady_state() -- one
sparse linear solve per candidate, no time integration. It therefore needs
no dt, no total_steps and no safety_factor, and run_stage2_optimizer() is
passed none of them. This keeps the Stage 2 hook as free of
SolverSection.method/.dt as the rest of this file (see the interface note
below). Its wall-clock is reported separately as TimingReport.optimize_s
rather than being folded into another phase.

IMPORTANT INTERFACE NOTE (fix for a prior bug report):
This file never reads SolverSection.method or SolverSection.dt -- those are
optional fields on the PipelineConfig dataclass used elsewhere in this
codebase (e.g. for Crank-Nicolson / steady-state runs driven from a config
file), but a stale/older local copy of config_loader.py may not define them.
To make main.py robust regardless of which config_loader.py version is on
disk, this file drives solver execution entirely from plain local variables
(dt, total_steps, safety_factor) captured directly from user input, and
only ever reads SolverSection.safety_factor -- a field that has existed on
every version of this dataclass -- via getattr() with a hardcoded fallback
as an extra safety margin.

STAGE 2 DIE-SIZE NOTE:
Stage 2's chiplet footprints are PHYSICAL (micrometers, see
src.optimizer.ChipletSpec) and do not shrink when the mesh is refined, so the
die must be large enough in real terms: nx*dx by ny*dy. The default catalog is
192 mm^2 of realistic blocks, which does NOT fit on this file's default grid
(60x60 at 25 um = a 1.5 x 1.5 mm die). Passing --optimize with a die that is
too small, or with a grid that would push Stage 2 onto the non-converging
iterative solver path, exits EXIT_PREFLIGHT_FAILED (2) BEFORE any solving,
with a concrete suggested command. Exit 2 therefore means "this configuration
cannot work", distinct from exit 1 for a genuine runtime failure.

Usage:
    python main.py                     # interactive prompts (default)
    python main.py --config foo.json   # load parameters from a JSON file instead
    python main.py --nx 80 --ny 80 --nz 50 --die-power-w 20   # non-interactive, explicit flags

    # Stage 1 + Stage 2 in one run, on a 19.6 x 19.6 mm package. nz=12 keeps
    # the grid at 19,200 unknowns (direct sparse path); at nz=24 this mesh
    # falls to the iterative backend and does not converge.
    python main.py --optimize --nx 40 --ny 40 --nz 12
                   --dx-um 490 --dy-um 490 --dz-um 40
    (one command; wrapped here for width)
"""

from __future__ import annotations

import argparse
import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import Optional

# Strict, package-qualified imports only -- main.py is always run from the
# repository root (`python main.py`), so `src` must resolve as a package.
from src.config_loader import (
    PipelineConfig, ConfigError, load_config,
    GridSection, BoundaryConditionsSection, StackSection, PowerSection,
    SolverSection, VisualizationSection,
)
from src.grid_builder import GridBuilder, GridBuilderError
from src.thermal_solver import (
    ThermalSolver, ThermalSolverError, DIRECT_SOLVE_MAX_UNKNOWNS,
)
from src.visualization import ThermalVisualizer, VisualizationError
from src.optimizer import (
    ChipletPlacementOptimizer, OptimizerError,
    DEFAULT_CHIPLET_CATALOG, render_optimization_report,
)


class PipelineError(Exception):
    """Raised on any unrecoverable failure in the end-to-end pipeline."""


@dataclass
class TimingReport:
    grid_build_s: float = 0.0
    solver_init_s: float = 0.0
    cfl_check_s: float = 0.0
    solve_s: float = 0.0
    visualization_s: float = 0.0
    # Stage 2 (optional). Stays 0.0 when the optimizer is not run, so it
    # contributes nothing to total_s on a plain Stage 1 invocation. When it
    # IS run it is counted here explicitly rather than silently inflating the
    # wall-clock of some other phase.
    optimize_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.grid_build_s + self.solver_init_s + self.cfl_check_s
                + self.solve_s + self.visualization_s + self.optimize_s)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("RISC3D-AI PIPELINE TIMING SUMMARY")
        print("=" * 60)
        rows = [
            ("Grid construction (GridBuilder)", self.grid_build_s),
            ("Solver initialization", self.solver_init_s),
            ("CFL stability check", self.cfl_check_s),
            ("Transient solve (explicit, N steps)", self.solve_s),
            ("WebGL 3D export + browser launch", self.visualization_s),
        ]
        if self.optimize_s > 0.0:
            rows.append(("Stage 2 placement optimization", self.optimize_s))
        for label, seconds in rows:
            print(f"  {label:<38s} {seconds:>10.4f} s")
        print("-" * 60)
        print(f"  {'TOTAL WALL-CLOCK TIME':<38s} {self.total_s:>10.4f} s")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Interactive terminal prompts
# ---------------------------------------------------------------------------

def prompt_float(label: str, default: float) -> float:
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"    Not a valid number -- using default {default}")
        return default


def prompt_int(label: str, default: int) -> int:
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"    Not a valid integer -- using default {default}")
        return default


def prompt_yes_no(label: str, default: bool = False) -> bool:
    """Yes/no prompt defaulting to `default` on a bare Enter.

    Returns `default` rather than raising if stdin is closed or the user
    interrupts, so a piped/non-interactive invocation that reaches this
    prompt degrades to the default instead of crashing the pipeline after
    Stage 1 has already produced its deliverable.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"  {label} {suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if raw == "":
        return default
    return raw in ("y", "yes")


def run_interactive_prompts():
    """
    Prompts for every parameter listed in the interactive-CLI requirement,
    in micrometers/Celsius/Watts as the user naturally thinks in them, then
    converts into the SI-unit PipelineConfig this codebase uses internally.
    Everything not explicitly prompted for (stack layer fractions, die
    footprint fraction, output directory, hotspot threshold) uses a fixed,
    documented default -- kept out of the prompt list per the requested
    parameter set, but easy to expose later if needed.
    """
    print("=" * 60)
    print("RISC3D-AI :: Interactive Simulation Setup")
    print("Press Enter to accept the bracketed default for any parameter.")
    print("=" * 60)

    print("\nGrid dimensions (voxel counts):")
    nx = prompt_int("nx", 60)
    ny = prompt_int("ny", 60)
    nz = prompt_int("nz", 40)

    print("\nVoxel spacing (micrometers):")
    dx_um = prompt_float("dx (um)", 25.0)
    dy_um = prompt_float("dy (um)", 25.0)
    dz_um = prompt_float("dz (um)", 10.0)

    print("\nThermal boundary conditions:")
    t_ambient_c = prompt_float("Ambient temperature (C)", 25.0)
    h_convective = prompt_float("Heat transfer coefficient h_convective (W/m^2K)", 1000.0)

    print("\nActive power source:")
    die_power_w = prompt_float("Active die total power (W)", 12.0)

    print("\nSolve parameters:")
    total_steps = prompt_int("Total simulation steps", 100)
    safety_factor = prompt_float("Safety factor (0-1]", 0.8)

    print()

    grid = GridSection(nx=nx, ny=ny, nz=nz, dx=dx_um * 1e-6, dy=dy_um * 1e-6, dz=dz_um * 1e-6)
    bc = BoundaryConditionsSection(t_ambient_c=t_ambient_c, h_convective=h_convective, bottom_bc="robin")
    stack = StackSection()      # fixed defaults: interposer_frac=0.40, bump_frac=0.15
    power = PowerSection(die_power_w=die_power_w)  # fixed default die_footprint_frac=0.5
    solver = SolverSection(safety_factor=safety_factor)  # total_steps handled locally, not stored here
    viz = VisualizationSection()  # output_dir="outputs", hotspot_threshold_c=85.0

    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()

    # total_steps is intentionally NOT a PipelineConfig/SolverSection field
    # (avoids ever depending on a field name that might not exist on a
    # stale local config_loader.py) -- it is threaded through main()
    # separately as a plain local variable.
    return config, total_steps


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RISC3D-AI end-to-end 3D IC thermal simulation pipeline. "
                     "Running with NO arguments launches interactive terminal prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Load parameters from a JSON config file instead of prompting "
                              "or using flag defaults below.")
    parser.add_argument("--nx", type=int, default=60)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dx-um", type=float, default=25.0, help="Voxel size along X (micrometers).")
    parser.add_argument("--dy-um", type=float, default=25.0, help="Voxel size along Y (micrometers).")
    parser.add_argument("--dz-um", type=float, default=10.0, help="Voxel size along Z (micrometers).")
    parser.add_argument("--t-ambient-c", type=float, default=25.0, help="Ambient temperature (degC).")
    parser.add_argument("--h-convective", type=float, default=1000.0, help="Convective coefficient (W/m^2K).")
    parser.add_argument("--die-power-w", type=float, default=12.0, help="Active die total power (W).")
    parser.add_argument("--total-steps", type=int, default=100, help="Total explicit simulation steps.")
    parser.add_argument("--safety-factor", type=float, default=0.8, help="CFL safety factor in (0, 1].")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory.")
    parser.add_argument("--no-browser", action="store_true",
                         help="Do not automatically open the generated HTML in a browser.")
    parser.add_argument("--optimize", action="store_true",
                         help="After Stage 1, run the Stage 2 automated chiplet placement "
                              "optimizer and write outputs/optimized_thermal_report.html.")
    parser.add_argument("--optimize-iterations", type=int, default=30,
                         help="Simulated-annealing proposals for Stage 2 (--optimize only).")
    parser.add_argument("--optimize-seed", type=int, default=0,
                         help="Random seed for the Stage 2 annealer (--optimize only).")
    return parser


def build_config_from_flags(args: argparse.Namespace):
    grid = GridSection(nx=args.nx, ny=args.ny, nz=args.nz,
                        dx=args.dx_um * 1e-6, dy=args.dy_um * 1e-6, dz=args.dz_um * 1e-6)
    bc = BoundaryConditionsSection(t_ambient_c=args.t_ambient_c, h_convective=args.h_convective, bottom_bc="robin")
    stack = StackSection()
    power = PowerSection(die_power_w=args.die_power_w)
    solver = SolverSection(safety_factor=args.safety_factor)
    viz = VisualizationSection(output_dir=args.output_dir)
    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()
    return config, args.total_steps


EXIT_PREFLIGHT_FAILED = 2


def preflight_optimizer_fit(config: PipelineConfig, catalog=DEFAULT_CHIPLET_CATALOG):
    """
    Checks BEFORE any solving that the Stage 2 chiplet catalog physically fits
    on the die the grid describes. Returns None if it fits, or a multi-line
    actionable error message if it does not.

    Why this runs up front: chiplet footprints are authored in micrometers
    (src.optimizer.ChipletSpec) while the die size is nx*dx by ny*dy, so a
    grid that is fine for Stage 1 can be far too small in PHYSICAL terms for
    a realistic catalog. The default catalog is 192 mm^2 of real blocks, but
    this file's default grid (60x60 at 25 um) is a 1.5 x 1.5 mm die -- only
    2.25 mm^2. Without this check the user pays for a full Stage 1 solve and
    only then discovers Stage 2 cannot run.

    Deliberately reads only the public physical properties of ChipletSpec
    (width_um / depth_um / area_mm2) and does not construct an optimizer, so
    it stays cheap and does not duplicate the optimizer's internals. The
    optimizer performs the authoritative check again in its own __init__.
    """
    import math as _math

    die_w_um = config.grid.nx * config.grid.dx * 1e6
    die_h_um = config.grid.ny * config.grid.dy * 1e6
    die_area_mm2 = (die_w_um * die_h_um) * 1e-6
    catalog_area_mm2 = sum(c.area_mm2 for c in catalog)
    n_unknowns = config.grid.nx * config.grid.ny * config.grid.nz

    # Constraint 1: the catalog must physically fit on the die.
    too_big = [c for c in catalog if c.width_um > die_w_um or c.depth_um > die_h_um]
    fits_by_area = catalog_area_mm2 <= die_area_mm2

    # Constraint 2: the grid must stay on the direct sparse solve path. A
    # package-scale die needs a coarse lateral pitch while the stack still
    # needs a fine dz, and on the resulting anisotropic mesh the iterative
    # backend does not converge (measured; see src/optimizer.py's SOLVER
    # BACKEND LIMIT note). Above the threshold Stage 2 fails mid-anneal, so
    # catch it here rather than after a full Stage 1 solve.
    aspect = max(config.grid.dx, config.grid.dy) / config.grid.dz if config.grid.dz > 0 else 0.0
    over_budget = n_unknowns > DIRECT_SOLVE_MAX_UNKNOWNS and aspect > 2.0

    if not too_big and fits_by_area and not over_budget:
        return None

    # --- Build one recommendation that satisfies BOTH constraints ---
    # Target a die at least 2x the catalog area, so the annealer has room to
    # move rather than being packed solid, and at least as large as the
    # biggest single block.
    needed_side_um = max(
        _math.sqrt(2.0 * catalog_area_mm2 * 1e6),
        max(c.width_um for c in catalog),
        max(c.depth_um for c in catalog),
    )
    # Choose the largest square lateral grid whose unknown count leaves room
    # for a usable stack depth, then derive the pitch from the die size.
    rec_nz = 12
    rec_n = int(_math.sqrt(DIRECT_SOLVE_MAX_UNKNOWNS / rec_nz))
    rec_pitch_um = _math.ceil(needed_side_um / rec_n / 10.0) * 10.0
    rec_dz_um = 40.0

    lines = ["Stage 2 cannot run with this configuration."]
    if too_big or not fits_by_area:
        lines += [
            "",
            "  The chiplet catalog does not fit on this die:",
            f"    Die:     {die_w_um:,.0f} x {die_h_um:,.0f} um = {die_area_mm2:.2f} mm^2 "
            f"({config.grid.nx}x{config.grid.ny} voxels at dx={config.grid.dx*1e6:.1f} "
            f"dy={config.grid.dy*1e6:.1f} um)",
            f"    Catalog: {catalog_area_mm2:.2f} mm^2 across {len(catalog)} blocks",
        ]
        for c in too_big:
            lines.append(f"      - '{c.name}' is {c.width_um:,.0f} x {c.depth_um:,.0f} um "
                         f"and exceeds the die on its own")
        if not too_big and not fits_by_area:
            lines.append("      - each block fits individually, but not all together")
        lines += [
            "    Chiplet footprints are PHYSICAL (micrometers) and do not shrink when",
            "    the mesh is refined, so the die must be enlarged via nx*dx and ny*dy.",
        ]
    if over_budget:
        lines += [
            "",
            f"  The grid is {n_unknowns:,} unknowns, above "
            f"DIRECT_SOLVE_MAX_UNKNOWNS ({DIRECT_SOLVE_MAX_UNKNOWNS:,}), on a mesh with",
            f"  aspect ratio {aspect:.1f}:1. Stage 2 would fall to the iterative backend,",
            "  which does not converge on meshes this anisotropic.",
        ]
    lines += [
        "",
        "  A configuration satisfying both constraints:",
        f"    python main.py --optimize --nx {rec_n} --ny {rec_n} --nz {rec_nz} "
        f"--dx-um {rec_pitch_um:.0f} --dy-um {rec_pitch_um:.0f} --dz-um {rec_dz_um:.0f}",
        f"    ({rec_n}x{rec_n} at {rec_pitch_um:.0f} um = "
        f"{rec_n * rec_pitch_um / 1000.0:.1f} x {rec_n * rec_pitch_um / 1000.0:.1f} mm die, "
        f"{rec_n * rec_n * rec_nz:,} unknowns)",
        "",
        "  Or supply a smaller catalog to ChipletPlacementOptimizer directly.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def build_chiplet_stack(config: PipelineConfig) -> GridBuilder:
    """
    Constructs a 3-region CoWoS-style vertical stack:
      z-layer 0 .. z_interposer_top      : passive Si interposer (bottom)
      z_interposer_top .. z_bump_top     : microbump + underfill composite (anisotropic)
      z_bump_top .. nz                   : Si compute die (top), with an active
                                            power-density footprint in its center
    Bottom face (z=0, interposer underside) is the heat-sink boundary
    (Robin, per this file's fixed bottom_bc="robin"); top and all sides
    are adiabatic.
    """
    try:
        gb = GridBuilder(config=config.to_dict())
    except GridBuilderError as exc:
        raise PipelineError(f"GridBuilder construction failed: {exc}") from exc

    nz = config.grid.nz
    nx, ny = config.grid.nx, config.grid.ny
    # Layer interfaces come from PHYSICAL thicknesses (um) resolved against
    # this mesh's dz, not from fractions of nz. See StackSection's docstring:
    # fraction authoring made the composite layer's realized thickness a
    # rounding artifact of nz, swinging -16.7%..+25.0% and moving the peak
    # temperature with it.
    z_interposer_end, z_bump_end = config.stack.resolve_layer_boundaries(nz, config.grid.dz)
    realized = config.stack.realized_thicknesses_um(nz, config.grid.dz)

    try:
        gb.assign_layer(slice(0, z_interposer_end), "silicon")
        gb.assign_layer(slice(z_interposer_end, z_bump_end), "bump_underfill_composite")
        gb.assign_layer(slice(z_bump_end, nz), "silicon")
        gb.tag_boundaries()

        margin_x = int(round(nx * (1.0 - config.power.die_footprint_frac) / 2.0))
        margin_y = int(round(ny * (1.0 - config.power.die_footprint_frac) / 2.0))
        x_slice = slice(margin_x, nx - margin_x)
        y_slice = slice(margin_y, ny - margin_y)
        z_slice = slice(z_bump_end, nz)

        gb.add_power_source(power_watts=config.power.die_power_w, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
        gb.validate_grid()
    except GridBuilderError as exc:
        raise PipelineError(f"Chiplet stack assembly failed: {exc}") from exc

    # Print REALIZED thicknesses next to the authored ones. A refinement study
    # is only meaningful if the geometry is identical at every level, so any
    # discretization-induced geometry change must be visible in the log rather
    # than hidden inside the result.
    print(
        f"[main] Stack assembled: interposer z[0:{z_interposer_end}], "
        f"bump+underfill z[{z_interposer_end}:{z_bump_end}], "
        f"Si die z[{z_bump_end}:{nz}] with {config.power.die_power_w:.2f}W active source "
        f"over footprint x[{margin_x}:{nx - margin_x}] y[{margin_y}:{ny - margin_y}]"
    )
    print(
        f"[main] Layer thicknesses (authored -> realized at dz={config.grid.dz*1e6:.2f}um): "
        f"interposer {config.stack.interposer_um:.1f} -> {realized['interposer_um']:.1f}um, "
        f"composite {config.stack.bump_um:.1f} -> {realized['bump_um']:.1f}um, "
        f"die {config.stack.die_um:.1f} -> {realized['die_um']:.1f}um"
    )
    for label, authored, got in (
        ("interposer", config.stack.interposer_um, realized["interposer_um"]),
        ("composite", config.stack.bump_um, realized["bump_um"]),
        ("die", config.stack.die_um, realized["die_um"]),
    ):
        if abs(got - authored) > 1e-9:
            print(f"[main] WARNING: {label} layer discretized to {got:.1f}um, "
                  f"{100.0*(got-authored)/authored:+.1f}% off the authored {authored:.1f}um. "
                  f"Choose dz so {authored:.1f}um is a whole number of cells.")
    return gb


def _open_in_browser(html_path: str, open_browser: bool, label: str) -> None:
    """Shared browser-launch helper: absolutize, open, and degrade gracefully.

    A missing or unavailable browser must never fail the pipeline -- the HTML
    file is the complete deliverable either way, and only the auto-launch
    convenience was lost. Mirrors the behaviour Stage 1 has always had for
    thermal_3d.html, now shared with Stage 2's report so the two cannot drift.
    """
    import os
    abs_path = os.path.abspath(html_path)
    if not open_browser:
        print(f"[main] Browser launch skipped (--no-browser). Open manually: {html_path}")
        return
    try:
        webbrowser.open(f"file://{abs_path}")
        print(f"[main] Opened {abs_path} in the default browser.")
    except Exception as browser_exc:  # noqa: BLE001
        print(f"[main] Could not auto-launch a browser for {label} ({browser_exc}). "
              f"Open manually: {abs_path}")


def print_stage1_results(config: PipelineConfig, report: dict) -> None:
    """Prints the Stage 1 thermal findings block."""
    print("\n" + "=" * 60)
    print("THERMAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Peak temperature:      {report['peak_temperature_c']:.2f} °C")
    print(f"  Peak location (i,j,k): {report['peak_location_ijk']}")
    print(f"  Peak location (um):    "
          f"({report['peak_location_um'][0]:.1f}, {report['peak_location_um'][1]:.1f}, "
          f"{report['peak_location_um'][2]:.1f})")
    print(f"  Voxels > {config.visualization.hotspot_threshold_c:.1f}°C:       {report['hotspot_count']}")
    print(f"  Interactive 3D view:   {report['html_path']}")
    print("=" * 60)


def run_stage2_optimizer(
    config: PipelineConfig,
    open_browser: bool,
    iterations: int,
    seed: int,
) -> dict:
    """
    Stage 2: automated chiplet placement optimization.

    Reuses the grid, boundary-condition and stack parameters run_pipeline()
    already gathered -- nothing is re-prompted. Deliberately passes NO timing
    parameters: the optimizer scores candidates with
    ThermalSolver.solve_steady_state(), a single linear solve per candidate,
    so dt / total_steps / safety_factor are meaningless to it. That also
    keeps this hook clear of SolverSection.method/.dt, exactly as the rest of
    main.py is (see the interface note in the module docstring).
    """
    print("\n" + "=" * 60)
    print("STAGE 2 :: AUTOMATED CHIPLET PLACEMENT OPTIMIZATION")
    print("=" * 60)
    total_power = sum(c.power_w for c in DEFAULT_CHIPLET_CATALOG)
    print(f"[main] Catalog: {len(DEFAULT_CHIPLET_CATALOG)} chiplets "
          f"({', '.join(c.name for c in DEFAULT_CHIPLET_CATALOG)}) totalling {total_power:.1f} W")
    print(f"[main] Objective: steady-state peak temperature "
          f"({iterations} annealing proposals, seed={seed})")

    try:
        optimizer = ChipletPlacementOptimizer.from_pipeline_config(
            config, catalog=DEFAULT_CHIPLET_CATALOG, seed=seed,
        )
        result = optimizer.optimize(iterations=iterations, verbose=True)
        report_path = render_optimization_report(
            result, output_dir=config.visualization.output_dir,
        )
    except OptimizerError as exc:
        raise PipelineError(f"Stage 2 placement optimization failed: {exc}") from exc

    print("\n" + result.summarize())
    print(f"  Optimization report:   {report_path}")

    _open_in_browser(report_path, open_browser, "the optimization report")
    return {"result": result, "report_path": report_path}


def run_pipeline(
    config: PipelineConfig,
    total_steps: int,
    open_browser: bool,
    run_optimizer: bool = False,
    prompt_for_optimizer: bool = False,
    optimize_iterations: int = 30,
    optimize_seed: int = 0,
) -> dict:
    timing = TimingReport()

    # --- Phase 1: Grid construction ---
    t0 = time.perf_counter()
    gb = build_chiplet_stack(config)
    timing.grid_build_s = time.perf_counter() - t0

    # --- Phase 2: Solver initialization ---
    t0 = time.perf_counter()
    try:
        solver = ThermalSolver.from_grid_builder(gb)
    except ThermalSolverError as exc:
        raise PipelineError(f"ThermalSolver initialization failed: {exc}") from exc
    timing.solver_init_s = time.perf_counter() - t0

    # --- Phase 3: Automated CFL stability check ---
    # NOTE: reads ONLY SolverSection.safety_factor (via getattr with a
    # hardcoded fallback for extra defensiveness against a stale/older
    # local config_loader.py). Never reads .method or .dt.
    safety_factor = getattr(config.solver, "safety_factor", 0.8)
    t0 = time.perf_counter()
    try:
        dt_stable = solver.compute_max_dt(safety_factor=safety_factor)
    except ThermalSolverError as exc:
        raise PipelineError(f"CFL stability check failed: {exc}") from exc
    timing.cfl_check_s = time.perf_counter() - t0

    total_time_s = dt_stable * total_steps
    print(
        f"[main] Automated CFL check: dt_stable={dt_stable:.6e}s "
        f"(safety_factor={safety_factor}) -> running exactly {total_steps} explicit steps "
        f"(total_time_s={total_time_s:.6e}s)"
    )

    # --- Phase 4: Transient solve for exactly `total_steps` explicit steps ---
    t0 = time.perf_counter()
    try:
        result = solver.run(
            total_time_s=total_time_s,
            dt=dt_stable,
            safety_factor=safety_factor,
            record_interval_s=None,
            max_steps=total_steps + 10,
        )
    except ThermalSolverError as exc:
        raise PipelineError(f"Transient solve failed: {exc}") from exc
    timing.solve_s = time.perf_counter() - t0

    print(f"[main] Solve complete: {result['steps_taken']} steps, final_time={result['final_time_s']:.6e}s")

    # --- Phase 5: Interactive WebGL 3D export (replaces static PNG output) ---
    t0 = time.perf_counter()
    try:
        viz = ThermalVisualizer(
            temperature_k=result["final_T"],
            dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
            output_dir=config.visualization.output_dir,
        )
        html_path = viz.render_interactive_webgl(filename="thermal_3d.html")
        hotspot_report = viz.get_hotspot_report(threshold_c=config.visualization.hotspot_threshold_c)

        _open_in_browser(html_path, open_browser, "the 3D thermal view")

        report = {
            "html_path": html_path,
            "hotspot_count": hotspot_report.count,
            "peak_temperature_c": hotspot_report.peak_temperature_c,
            "peak_location_um": hotspot_report.peak_location_um,
            "peak_location_ijk": hotspot_report.peak_location_ijk,
        }
    except VisualizationError as exc:
        raise PipelineError(f"WebGL visualization export failed: {exc}") from exc
    timing.visualization_s = time.perf_counter() - t0

    # --- Phase 6 (optional): Stage 2 automated placement optimization ---
    # Offered only AFTER Stage 1 has fully completed and written its
    # deliverable, so declining (or a failure here) never costs the user the
    # thermal_3d.html they already earned.
    # Stage 1's findings are printed BEFORE the Stage 2 offer, so an
    # interactive user decides whether to optimize having already seen how
    # hot the current placement actually runs.
    print_stage1_results(config, report)

    optimizer_outcome = None
    should_optimize = run_optimizer
    if not should_optimize and prompt_for_optimizer:
        print()
        should_optimize = prompt_yes_no(
            "Run Stage 2 automated chiplet placement optimizer?", default=False
        )

    if should_optimize:
        t0 = time.perf_counter()
        optimizer_outcome = run_stage2_optimizer(
            config, open_browser=open_browser,
            iterations=optimize_iterations, seed=optimize_seed,
        )
        timing.optimize_s = time.perf_counter() - t0

    return {
        "timing": timing,
        "solve_result": result,
        "viz_report": report,
        "optimizer": optimizer_outcome,
    }


def main(argv=None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # True only on the fully interactive path, which is the one that may ask
    # about Stage 2 after Stage 1 finishes. A flag- or config-driven run is
    # non-interactive by definition and must never block on a prompt.
    interactive = False

    try:
        if args.config is not None:
            config = load_config(args.config)
            total_steps = args.total_steps
        elif argv is None and len(sys.argv) == 1:
            # `python main.py` with truly no arguments -> interactive prompts.
            config, total_steps = run_interactive_prompts()
            interactive = True
        else:
            # Any explicit flag supplied -> non-interactive, flag-driven run.
            config, total_steps = build_config_from_flags(args)
    except ConfigError as exc:
        print(f"\n[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n[main] Input cancelled. Exiting.")
        return 1

    print("=" * 60)
    print("RISC3D-AI :: 3D IC Thermal Simulation Pipeline")
    print("=" * 60)
    print(f"Grid: {config.grid.nx}x{config.grid.ny}x{config.grid.nz} voxels "
          f"({config.grid.dx*1e6:.1f}x{config.grid.dy*1e6:.1f}x{config.grid.dz*1e6:.1f} um)")
    print(f"Ambient: {config.boundary_conditions.t_ambient_c:.1f}C | "
          f"h_convective: {config.boundary_conditions.h_convective:.1f} W/m^2K | "
          f"Die power: {config.power.die_power_w:.2f}W | Steps: {total_steps}")
    print("=" * 60)

    if args.optimize_iterations < 1:
        print(f"\n[FATAL] --optimize-iterations must be >= 1, got {args.optimize_iterations}",
              file=sys.stderr)
        return 1

    # Pre-flight: if Stage 2 was explicitly requested via --optimize, verify
    # the catalog fits BEFORE paying for a full Stage 1 solve. Exits 2 (not 1)
    # so a caller can distinguish "your configuration cannot work" from a
    # genuine runtime failure. The interactive prompt path is not pre-checked
    # here -- that user has not asked for Stage 2 yet, and Stage 1 is still
    # worth running for them.
    if args.optimize:
        preflight_error = preflight_optimizer_fit(config)
        if preflight_error is not None:
            print(f"\n[FATAL] {preflight_error}", file=sys.stderr)
            return EXIT_PREFLIGHT_FAILED

    try:
        outcome = run_pipeline(
            config, total_steps,
            open_browser=not args.no_browser,
            run_optimizer=args.optimize,
            prompt_for_optimizer=interactive,
            optimize_iterations=args.optimize_iterations,
            optimize_seed=args.optimize_seed,
        )
    except PipelineError as exc:
        print(f"\n[FATAL] Pipeline aborted: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - final safety net, always exit cleanly
        print(f"\n[FATAL] Unexpected error: {exc}", file=sys.stderr)
        return 1

    # Stage 1's results block is printed inside run_pipeline(), ahead of the
    # Stage 2 offer, so it is not repeated here.
    outcome["timing"].print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())