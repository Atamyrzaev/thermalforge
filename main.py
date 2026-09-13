#!/usr/bin/env python3
"""
main.py
RISC3D-AI :: End-to-End 3D IC Thermal Simulation Pipeline (Interactive CLI)

Entry point that:
  1. Prompts the user directly in the terminal for all simulation parameters
     (grid size, spacing, ambient temp, cooling preset, convective
     coefficient, die power, safety factor) -- NO configs/config.json
     dependency for the default interactive workflow. Press Enter on any
     prompt to accept the bracketed default.
  2. Configures a multi-layer TSMC-style advanced-packaging chiplet stack
     (Interposer -> Microbump+Underfill composite -> Silicon Die with an
     active compute-block power density Q) via src.grid_builder.GridBuilder.
  3. Constructs src.thermal_solver.ThermalSolver and solves the steady state
     directly (one sparse linear solve). A transient is opt-in via
     --transient <ms>, time-stepped at the automated CFL-stable dt.
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
192 mm^2 of realistic blocks; the default grid (40x40 at 500 um) is a 20 x 20
mm die and holds it at 48% fill. Passing --optimize with a die that is too
small exits EXIT_PREFLIGHT_FAILED (2) BEFORE any solving, with a concrete
suggested command. Exit 2 therefore means "this configuration cannot work",
distinct from exit 1 for a genuine runtime failure.

COOLING REGIME NOTE:
Every absolute temperature here is set mostly by one input, the effective
sink coefficient h, and results at different h describe different products.
So the regime is always named: two presets (COOLING_PRESETS), and a header
that prints the heat flux in W/cm^2, the 1/(h*A) floor on R_ja and the
lateral spreading length lambda before anything is solved, then the measured
R_ja with the result. See the COOLING REGIMES block below for why lambda
decides whether Stage 2 placement is worth running at all.

Usage:
    python main.py                     # interactive prompts (default)
    python main.py --config foo.json   # load parameters from a JSON file instead
    python main.py --preset cowos-coldplate                      # 150 W, h = 25000
    python main.py --preset cowos-coldplate --h-convective 10000 # reported as 'custom'
    python main.py --die-power-w 20 --h-convective 5000          # custom regime

    # Stage 1 + Stage 2 in one run on the default 20 x 20 mm package
    python main.py --optimize --no-browser
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import webbrowser
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Strict, package-qualified imports only -- main.py is always run from the
# repository root (`python main.py`), so `src` must resolve as a package.
from src.config_loader import (
    PipelineConfig, ConfigError, load_config,
    GridSection, BoundaryConditionsSection, StackSection, PowerSection,
    SolverSection, VisualizationSection,
)
from src.grid_builder import GridBuilder, GridBuilderError, MATERIALS
from src.thermal_solver import (
    ThermalSolver, ThermalSolverError, DIRECT_SOLVE_MAX_UNKNOWNS,
)
from src.visualization import ThermalVisualizer, VisualizationError
from src.optimizer import (
    ChipletPlacementOptimizer, OptimizerError,
    DEFAULT_CHIPLET_CATALOG, render_optimization_report,
    scale_catalog_to_total_power,
)
from src.exhaustive import ExhaustivePlacementSearch, ExhaustiveSearchError
from src.validation_status import (
    STATUS as VALIDATION_STATUS,
    html_block as validation_html_block, text_block as validation_text_block,
)
from src.experiment import (
    CONSTRAINTS as EXPERIMENT_CONSTRAINTS, Experiment, ExperimentError,
    STACK_MATERIALS, build_stack,
)
from src.experiment_report import render_experiment_report


# A transient shorter than this many time constants cannot support a
# threshold or safety verdict. 3*tau is 95% of the steady rise; below that the
# field is still visibly moving. Used to gate the verdict in
# print_stage1_results() and to warn on --total-steps.
STEADY_STATE_TAU_MULTIPLE = 3.0


# ---------------------------------------------------------------------------
# COOLING REGIMES
# ---------------------------------------------------------------------------
# h is an EFFECTIVE coefficient on the die backside: TIM1, lid, TIM2 and the
# heatsink or cold plate lumped into one film, so 1/h is the whole area-
# specific resistance above the die (h = 1000 W/m^2K is 1000 mm^2K/W; h =
# 25000 is 40 mm^2K/W). It sets the absolute temperature more than anything
# else in the model, so it is never left implicit.
#
# It also decides whether placement matters. Heat spreads laterally over
# roughly lambda = sqrt(sum(k_xy * t) / h) before the sink removes it.
# Measured on the default 40x40x12 grid with the src/composite_materials.py
# conductivities, 4-block catalog scaled to P, exhaustive search on the
# 2000 um lattice:
#
#   preset            P      h       lambda   Stage 2: initial -> optimum    gain
#   passive-lowpower  12 W   1000    5.48 mm   83.23 ->  70.21 degC   13.02 K (22.4% of rise)
#   cowos-coldplate   150 W  25000   1.10 mm  105.40 -> 100.77 degC    4.63 K ( 5.8% of rise)
#
# With lambda comparable to the block spacing, blocks share heat and moving
# them apart pays. With lambda well under it, each block sees essentially
# only its own footprint -- the 150 W Stage 1 rise of 59.09 K is 1.5% under
# the local 1-D q''/h = 60 K -- and placement is a weak lever. Placement
# optimization is a design lever in weakly cooled packages, not strongly
# cooled ones.
@dataclass(frozen=True)
class CoolingPreset:
    name: str
    die_power_w: float
    h_convective: float     # effective W/m^2K, referenced to the die backside
    description: str


COOLING_PRESETS = {
    "passive-lowpower": CoolingPreset(
        name="passive-lowpower", die_power_w=12.0, h_convective=1000.0,
        description="low-power part, spreader to forced air; weakly cooled",
    ),
    "cowos-coldplate": CoolingPreset(
        name="cowos-coldplate", die_power_w=150.0, h_convective=25000.0,
        description="CoWoS-class part under a liquid cold plate; strongly cooled",
    ),
}
DEFAULT_COOLING_PRESET = "passive-lowpower"

# STACK_MATERIALS now lives in src/experiment.py, which assembles the same
# three regions for the experiment path. Imported above, so the legacy stack
# built here and the experiment stack built there cannot drift apart.


def _k_inplane(material: str) -> float:
    """In-plane conductivity sqrt(kx*ky) of an isotropic or anisotropic
    MATERIALS entry."""
    props = MATERIALS[material]
    kx = float(props.get("kx", props.get("k")))
    ky = float(props.get("ky", props.get("k")))
    return math.sqrt(kx * ky)


def active_footprint_slices(config: PipelineConfig):
    """(x_slice, y_slice) of the centred active power footprint.

    The one definition shared by build_chiplet_stack(), which puts the power
    there, and the regime header, which reports the heat flux over it -- so
    the W/cm^2 printed is over the area the solver actually heats.
    """
    nx, ny = config.grid.nx, config.grid.ny
    margin_x = int(round(nx * (1.0 - config.power.die_footprint_frac) / 2.0))
    margin_y = int(round(ny * (1.0 - config.power.die_footprint_frac) / 2.0))
    return slice(margin_x, nx - margin_x), slice(margin_y, ny - margin_y)


def regime_label(config: PipelineConfig) -> str:
    """The preset whose power AND h both match this config, else 'custom'.

    Derived from the values, not from which flag was typed, so an overridden
    preset can never be reported under the preset's name.
    """
    for preset in COOLING_PRESETS.values():
        if (math.isclose(config.power.die_power_w, preset.die_power_w, rel_tol=1e-12)
                and math.isclose(config.boundary_conditions.h_convective,
                                 preset.h_convective, rel_tol=1e-12)):
            return preset.name
    return "custom"


def regime_summary(config: PipelineConfig) -> dict:
    """The numbers that fix the cooling regime, all known before any solve."""
    g, bcs = config.grid, config.boundary_conditions
    if config.chiplets:
        # Chiplet mode: the active area is the chiplets' own footprints, which
        # is what actually carries the power, not the legacy centred rectangle.
        footprint_m2 = sum(c.width_um * c.height_um for c in config.chiplets) * 1e-12
    else:
        x_sl, y_sl = active_footprint_slices(config)
        footprint_m2 = (x_sl.stop - x_sl.start) * g.dx * (y_sl.stop - y_sl.start) * g.dy
    sink_m2 = g.nx * g.dx * g.ny * g.dy
    p = config.power.die_power_w
    h = bcs.h_convective
    robin = (bcs.bottom_bc if bcs.bottom_bc != "neumann" else bcs.top_bc) == "robin"
    # A Dirichlet sink pins its face at ambient: no film, a zero floor, and
    # no spreading length (the sink takes heat out wherever it arrives).
    r_conv = 1.0 / (h * sink_m2) if robin else 0.0
    thicknesses_m = (config.stack.interposer_um * 1e-6, config.stack.bump_um * 1e-6,
                     config.stack.die_um * 1e-6)
    sheet_w_per_k = sum(_k_inplane(m) * t for m, t in zip(STACK_MATERIALS, thicknesses_m))
    return {
        "label": regime_label(config),
        "footprint_label": ("the chiplet footprints" if config.chiplets
                            else "the active footprint"),
        "footprint_mm2": footprint_m2 * 1e6,
        "sink_mm2": sink_m2 * 1e6,
        "flux_footprint_w_cm2": p / (footprint_m2 * 1e4) if footprint_m2 > 0 else float("nan"),
        "flux_sink_w_cm2": p / (sink_m2 * 1e4),
        "r_conv_k_per_w": r_conv,
        "floor_rise_k": p * r_conv,
        "lambda_mm": math.sqrt(sheet_w_per_k / h) * 1e3 if robin else 0.0,
    }


def print_regime_header(config: PipelineConfig) -> dict:
    """Prints the cooling regime -- name, heat flux, R_ja floor, spreading
    length -- so no result is ever read without it. R_ja itself needs the
    solved peak and is printed with the results; what is known up front is
    its hard lower bound 1/(h*A), because the peak rise can never be below
    the mean sink-face rise and that is pinned to exactly P/(h*A)."""
    r = regime_summary(config)
    tag = "preset" if r["label"] != "custom" else "not a preset; compare with care"
    print(f"Cooling regime: {r['label']} ({tag})")
    print(f"Heat flux: {r['flux_footprint_w_cm2']:.2f} W/cm^2 over {r['footprint_label']} "
          f"({r['footprint_mm2']:.1f} mm^2) | {r['flux_sink_w_cm2']:.2f} W/cm^2 over the "
          f"sink face ({r['sink_mm2']:.1f} mm^2)")
    if r["r_conv_k_per_w"] > 0:
        print(f"R_ja floor: 1/(h*A_sink) = {r['r_conv_k_per_w']:.4f} K/W, so the peak rise "
              f"is at least P*R = {r['floor_rise_k']:.2f} K (measured R_ja follows the solve)")
        print(f"Spreading length: lambda = sqrt(sum(k_xy*t)/h) = {r['lambda_mm']:.2f} mm")
    else:
        print("R_ja floor: 0 (Dirichlet sink pinned at ambient; no film resistance)")
    return r


def estimate_tau_from_stack(stack, h_convective: float) -> float:
    """First-order package time constant, in seconds, from the AUTHORED stack
    and boundary coefficient alone -- no grid or solver required.

        tau = R*C = [1/(h*A)] * [sum_layers(rho*cp * thickness) * A]
                  = sum_layers(rho*cp * thickness) / h

    The area cancels, which is why this is computable before the user has
    picked a mesh. Needed so the duration prompt can display the timescale
    the entered number will be measured against.

    Agrees with ThermalSolver.estimate_time_constant() on the assembled grid
    to the accuracy of the layer discretization: measured 394.79 ms by both
    routes for the default 100/40/100 um stack at h = 1000 W/m^2K.
    """
    si = MATERIALS["silicon"]
    comp = MATERIALS["bump_underfill_composite"]
    rho_cp_si = si["rho"] * si["cp"]
    rho_cp_comp = comp["rho"] * comp["cp"]
    areal_c = (rho_cp_si * (stack.interposer_um + stack.die_um) * 1e-6
               + rho_cp_comp * stack.bump_um * 1e-6)
    if h_convective <= 0:
        raise PipelineError(f"h_convective must be > 0 to define a time constant, "
                            f"got {h_convective}")
    return areal_c / h_convective


class PipelineError(Exception):
    """Raised on any unrecoverable failure in the end-to-end pipeline."""


@dataclass
class TimingReport:
    grid_build_s: float = 0.0
    solver_init_s: float = 0.0
    cfl_check_s: float = 0.0
    solve_s: float = 0.0
    steady_state_s: float = 0.0
    visualization_s: float = 0.0
    # Stage 2 (optional). Stays 0.0 when the optimizer is not run, so it
    # contributes nothing to total_s on a plain Stage 1 invocation. When it
    # IS run it is counted here explicitly rather than silently inflating the
    # wall-clock of some other phase.
    optimize_s: float = 0.0

    @property
    def total_s(self) -> float:
        return (self.grid_build_s + self.solver_init_s + self.cfl_check_s
                + self.solve_s + self.steady_state_s + self.visualization_s
                + self.optimize_s)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("RISC3D-AI PIPELINE TIMING SUMMARY")
        print("=" * 60)
        rows = [
            ("Grid construction (GridBuilder)", self.grid_build_s),
            ("Solver initialization", self.solver_init_s),
            ("CFL stability check", self.cfl_check_s),
            ("Transient solve (explicit, N steps)", self.solve_s),
            ("Steady-state solve (direct)", self.steady_state_s),
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
    nx = prompt_int("nx", 40)
    ny = prompt_int("ny", 40)
    nz = prompt_int("nz", 12)

    print("\nVoxel spacing (micrometers):")
    dx_um = prompt_float("dx (um)", 500.0)
    dy_um = prompt_float("dy (um)", 500.0)
    dz_um = prompt_float("dz (um)", 20.0)

    print("\nCooling regime:")
    names = list(COOLING_PRESETS)
    for i, name in enumerate(names, start=1):
        p = COOLING_PRESETS[name]
        print(f"  {i}) {name:<17} {p.die_power_w:6.1f} W at h = {p.h_convective:>6,.0f} "
              f"W/m^2K -- {p.description}")
    default_choice = names.index(DEFAULT_COOLING_PRESET) + 1
    choice = prompt_int("Preset number (its values become the defaults below)", default_choice)
    if not 1 <= choice <= len(names):
        print(f"    No preset {choice} -- using {DEFAULT_COOLING_PRESET}")
        choice = default_choice
    preset = COOLING_PRESETS[names[choice - 1]]

    print("\nThermal boundary conditions:")
    t_ambient_c = prompt_float("Ambient temperature (C)", 25.0)
    h_convective = prompt_float("Heat transfer coefficient h_convective (W/m^2K)",
                                preset.h_convective)

    print("\nActive power source:")
    die_power_w = prompt_float("Active die total power (W)", preset.die_power_w)

    print("\nSolve parameters:")
    # No duration prompt. The default run solves the steady state directly --
    # one linear solve, no time-stepping -- and that is the field every
    # threshold verdict uses. Asking for a transient duration up front made
    # the user pay 11,117 steps and 7.1 s to reach 4.88% of the steady rise,
    # for a number the verdict then explicitly discarded. A transient is now
    # opt-in via --transient <ms>, where it produces a peak-vs-time series
    # rather than a single uninterpretable instant.
    #
    # tau is still shown, because it is what makes any later --transient value
    # meaningful.
    tau_ms = estimate_tau_from_stack(StackSection(), h_convective) * 1e3
    print(f"  Package time constant tau ~ {tau_ms:.1f} ms.")
    print("  This run solves the STEADY STATE directly (no time-stepping).")
    print(f"  For a time series, re-run with --transient <ms> "
          f"(e.g. --transient {STEADY_STATE_TAU_MULTIPLE*tau_ms:.0f} to reach ~95%).")
    safety_factor = prompt_float("Safety factor (0-1]", 0.8)

    print()

    grid = GridSection(nx=nx, ny=ny, nz=nz, dx=dx_um * 1e-6, dy=dy_um * 1e-6, dz=dz_um * 1e-6)
    bc = BoundaryConditionsSection(t_ambient_c=t_ambient_c, h_convective=h_convective,
                                   bottom_bc="neumann", top_bc="robin")
    stack = StackSection()      # authored defaults: 100 / 40 / 100 um
    power = PowerSection(die_power_w=die_power_w)  # fixed default die_footprint_frac=0.5
    solver = SolverSection(safety_factor=safety_factor)  # total_steps handled locally, not stored here
    viz = VisualizationSection()  # output_dir="outputs", hotspot_threshold_c=85.0

    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()

    # Threaded through main() as plain locals, never as PipelineConfig fields
    # -- see the interface note in the module docstring. total_steps is None
    # on this path: the user gave a DURATION, and the step count is derived
    # from the CFL dt once the solver exists and dt is known.
    return config, None, None


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RISC3D-AI end-to-end 3D IC thermal simulation pipeline. "
                     "Running with NO arguments launches interactive terminal prompts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                         help="Load parameters from a JSON config file instead of prompting "
                              "or using flag defaults below.")
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--dx-um", type=float, default=500.0, help="Voxel size along X (micrometers).")
    parser.add_argument("--dy-um", type=float, default=500.0, help="Voxel size along Y (micrometers).")
    parser.add_argument("--dz-um", type=float, default=20.0, help="Voxel size along Z (micrometers).")
    parser.add_argument("--t-ambient-c", type=float, default=25.0, help="Ambient temperature (degC).")
    parser.add_argument("--preset", choices=tuple(COOLING_PRESETS), default=None,
                         help="Named cooling regime; sets --die-power-w and --h-convective "
                              "together. passive-lowpower = 12 W at h=1000 (used when neither "
                              "is given); cowos-coldplate = 150 W at h=25000. An explicit "
                              "--die-power-w or --h-convective overrides the preset's value, "
                              "and the header then reports the regime as 'custom'.")
    parser.add_argument("--h-convective", type=float, default=None,
                         help="Effective sink coefficient on the die backside (W/m^2K). "
                              "Default: the preset's value.")
    parser.add_argument("--die-power-w", type=float, default=None,
                         help="Active die total power (W). Default: the preset's value.")
    parser.add_argument("--transient", type=float, default=None, metavar="MS",
                         help="Also run a transient for this many milliseconds and record a "
                              "log-spaced peak-vs-time series. Omitted by default: the "
                              "steady state is one linear solve and is what every threshold "
                              "verdict uses, so time-stepping is opt-in.")
    parser.add_argument("--total-steps", type=int, default=None,
                         help="Explicit step count, for scripted runs. Prefer --transient, "
                              "which is expressed in physical time. Warns when the implied "
                              "duration is a small fraction of the package time constant.")
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
                         help="Random seed for the Stage 2 annealer (--optimize-method anneal only).")
    parser.add_argument("--optimize-method", choices=("exhaustive", "anneal"),
                         default="exhaustive",
                         help="Stage 2 search strategy. 'exhaustive' evaluates EVERY layout on "
                              "the placement lattice and returns the proven global optimum "
                              "(no randomness, no seed dependence). 'anneal' is the simulated-"
                              "annealing heuristic: faster on very fine lattices, but it samples "
                              "rather than proves and a different seed gives a different answer.")
    parser.add_argument("--no-scale-catalog-power", action="store_true",
                         help="Do NOT rescale the Stage 2 chiplet catalog to the Stage 1 "
                              "--die-power-w. By default the catalog (36.0 W as authored) is "
                              "scaled to match the die power the pipeline was given, "
                              "preserving the relative split, so both stages model the same "
                              "chip. Pass this to run the catalog at its authored wattage.")
    parser.add_argument("--placement-pitch-um", type=float, default=2000.0,
                         help="Placement lattice for Stage 2, in micrometers. Exhaustive cost "
                              "grows steeply as this shrinks (10x10 lattice = 8.3M layouts; "
                              "20x20 = 1.8 billion).")
    return parser


def build_config_from_flags(args: argparse.Namespace):
    grid = GridSection(nx=args.nx, ny=args.ny, nz=args.nz,
                        dx=args.dx_um * 1e-6, dy=args.dy_um * 1e-6, dz=args.dz_um * 1e-6)
    # Preset first, explicit flags over it. regime_label() later names the
    # result from the VALUES, so an overridden preset reports as 'custom'.
    preset = COOLING_PRESETS[args.preset or DEFAULT_COOLING_PRESET]
    h_convective = args.h_convective if args.h_convective is not None else preset.h_convective
    die_power_w = args.die_power_w if args.die_power_w is not None else preset.die_power_w
    bc = BoundaryConditionsSection(t_ambient_c=args.t_ambient_c, h_convective=h_convective,
                                   bottom_bc="neumann", top_bc="robin")
    stack = StackSection()
    power = PowerSection(die_power_w=die_power_w)
    solver = SolverSection(safety_factor=args.safety_factor)
    viz = VisualizationSection(output_dir=args.output_dir)
    config = PipelineConfig(grid=grid, boundary_conditions=bc, stack=stack, power=power,
                             solver=solver, visualization=viz)
    config.validate()
    # Flag path keeps --total-steps for scripted runs. duration_s is None
    # here, so run_pipeline() uses the step count directly (and warns if the
    # duration it implies is a small fraction of tau).
    return config, args.total_steps, None


EXIT_PREFLIGHT_FAILED = 2

# Flags that change WHAT is simulated. With --config the file is the single
# source of truth, so one of these beside it is refused rather than ignored --
# which is what used to happen: --h-convective 5000 --die-power-w 99 changed
# nothing and said nothing.
CONFIG_FILE_CONFLICTS = frozenset({
    "--preset", "--nx", "--ny", "--nz", "--dx-um", "--dy-um", "--dz-um",
    "--t-ambient-c", "--h-convective", "--die-power-w", "--safety-factor",
    "--no-scale-catalog-power",
})


def _explicit_flags(argv_list) -> frozenset:
    """The long options actually typed, so a flag that contradicts --config
    can be refused instead of silently dropped."""
    return frozenset(token.split("=", 1)[0] for token in argv_list
                     if isinstance(token, str) and token.startswith("--"))


def preflight_optimizer_fit(config: PipelineConfig, catalog=DEFAULT_CHIPLET_CATALOG):
    """
    Checks BEFORE any solving that the Stage 2 chiplet catalog physically fits
    on the die the grid describes. Returns None if it fits, or a multi-line
    actionable error message if it does not.

    Why this runs up front: chiplet footprints are authored in micrometers
    (src.optimizer.ChipletSpec) while the die size is nx*dx by ny*dy, so a
    grid that is fine for Stage 1 can be far too small in PHYSICAL terms for
    a realistic catalog. The default catalog is 192 mm^2 of real blocks; a
    60x60 grid at 25 um -- a reasonable Stage 1 mesh, and this file's default
    before the scale fix -- is a 1.5 x 1.5 mm die, only 2.25 mm^2. Without
    this check the user pays for a full Stage 1 solve and only then discovers
    Stage 2 cannot run.

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

    # NOTE: this function used to also refuse any grid above
    # DIRECT_SOLVE_MAX_UNKNOWNS with a mesh aspect ratio > 2:1, because the
    # steady-state solve then fell to a Jacobi-preconditioned BiCGSTAB that
    # did not converge on anisotropic meshes. That guard is GONE because the
    # underlying defect is fixed: solve_steady_state() now runs conjugate
    # gradients on S = -diag(rho_cp) @ M, which is exactly symmetric positive
    # definite, and converges at every size measured (40x40x24 at 50:1 aspect:
    # 3,062 iterations, peak 200.3900 degC, matching a direct factorization to
    # 4.8e-09 degC). Keeping the guard would now block configurations that
    # work, which is exactly what it did before this comment replaced it.
    #
    # Large grids are still SLOWER (CG iteration count grows with N), but slow
    # is a scheduling problem, not a correctness one, and every solve is
    # residual-verified inside solve_steady_state(). Only genuine
    # impossibilities are refused here.
    if not too_big and fits_by_area:
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
    # Keep the caller's z-discretization: it already holds the authored
    # stack exactly (validate() refuses any that does not). This used to
    # hardcode --nz 12 --dz-um 40, a 480 um mesh under the 240 um stack --
    # advice that the stack check now rejects. Then choose the largest square
    # lateral grid that stays on the direct path at that depth, and derive
    # the pitch from the die size.
    rec_nz = config.grid.nz
    rec_dz_um = config.grid.dz * 1e6
    rec_n = int(_math.sqrt(DIRECT_SOLVE_MAX_UNKNOWNS / rec_nz))
    rec_pitch_um = _math.ceil(needed_side_um / rec_n / 10.0) * 10.0

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
    lines += [
        "",
        "  A configuration satisfying both constraints:",
        f"    python main.py --optimize --nx {rec_n} --ny {rec_n} --nz {rec_nz} "
        f"--dx-um {rec_pitch_um:.0f} --dy-um {rec_pitch_um:.0f} --dz-um {rec_dz_um:g}",
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
    The heat sink is the TOP face (z = nz-1, the die backside), which is how a
    CoWoS part actually cools: die -> TIM1 -> lid -> TIM2 -> heatsink. The
    bottom face (interposer underside) and all four sides are adiabatic.
    Earlier versions sank through the bottom with an adiabatic top -- the
    package upside down. Which face is the sink comes from
    config.boundary_conditions.bottom_bc / top_bc, not from this function.
    """
    try:
        gb = build_stack(config)          # materials and boundary tags, no power
    except (GridBuilderError, ConfigError) as exc:
        raise PipelineError(f"GridBuilder construction failed: {exc}") from exc

    nz = config.grid.nz
    nx, ny = config.grid.nx, config.grid.ny
    # Layer interfaces come from PHYSICAL thicknesses (um) resolved against
    # this mesh's dz, not from fractions of nz. See StackSection's docstring:
    # fraction authoring made the composite layer's realized thickness a
    # rounding artifact of nz, swinging -16.7%..+25.0% and moving the peak
    # temperature with it.
    try:
        z_interposer_end, z_bump_end = config.stack.resolve_layer_boundaries(nz, config.grid.dz)
        realized = config.stack.realized_thicknesses_um(nz, config.grid.dz)
    except ConfigError as exc:
        raise PipelineError(f"The mesh cannot hold the authored stack: {exc}") from exc

    try:
        x_slice, y_slice = active_footprint_slices(config)
        z_slice = slice(z_bump_end, nz)

        gb.add_power_source(power_watts=config.power.die_power_w, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
        gb.validate_grid()
    except GridBuilderError as exc:
        raise PipelineError(f"Chiplet stack assembly failed: {exc}") from exc

    # Realized thicknesses next to the authored ones. They are equal by
    # construction -- StackSection.layer_cells() refuses any mesh where they
    # would differ -- and are printed so that is visible rather than assumed.
    # (A warning loop for a rounded or clamped layer used to follow; the
    # stack check makes that case unreachable, so it is gone.)
    print(
        f"[main] Stack assembled: interposer z[0:{z_interposer_end}], "
        f"bump+underfill z[{z_interposer_end}:{z_bump_end}], "
        f"Si die z[{z_bump_end}:{nz}] with {config.power.die_power_w:.2f}W active source "
        f"over footprint x[{x_slice.start}:{x_slice.stop}] y[{y_slice.start}:{y_slice.stop}]"
    )
    print(
        f"[main] Layer thicknesses (authored -> realized at dz={config.grid.dz*1e6:.2f}um): "
        f"interposer {config.stack.interposer_um:.1f} -> {realized['interposer_um']:.1f}um, "
        f"composite {config.stack.bump_um:.1f} -> {realized['bump_um']:.1f}um, "
        f"die {config.stack.die_um:.1f} -> {realized['die_um']:.1f}um"
    )
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
    """Prints the Stage 1 thermal findings block.

    Two temperatures are reported and never conflated: the transient peak,
    always labelled with the elapsed time it belongs to, and the steady-state
    peak, which is what the threshold verdict is judged against.

    The safety verdict is GATED. A transient that has covered a fraction of
    one time constant contains no information about whether a design is safe,
    and this block previously let the visualizer assert "within safe operating
    range" from exactly such a run -- a 0.1776 ms snapshot at 0.045% of tau,
    reporting ambient + 0.1 K. The verdict now comes from the steady state, and
    when the transient is short that is said explicitly rather than papered
    over.
    """
    threshold = config.visualization.hotspot_threshold_c
    tau_s = report["tau_s"]
    t_sim = report["transient_duration_s"]
    frac = report["converged_fraction"]
    steady_peak = report["peak_temperature_c"]
    has_transient = t_sim is not None

    print("\n" + "=" * 60)
    print("THERMAL ANALYSIS RESULTS")
    print("=" * 60)
    print(f"  Package time constant: {tau_s*1e3:.2f} ms")
    print()
    if has_transient:
        print(f"  TRANSIENT (t = {t_sim*1e3:.4f} ms = {t_sim/tau_s:.6f} tau, "
              f"{frac*100:.4f}% of steady rise)")
        print(f"    Peak at that instant: {report['transient_peak_temperature_c']:.2f} °C")
        if report.get("transient_html_path"):
            print(f"    Separate 3D view:     {report['transient_html_path']}")
        print()
    print("  STEADY STATE (t -> infinity, direct solve)")
    print(f"    Peak temperature:     {steady_peak:.2f} °C")
    print(f"    Peak location (i,j,k):{report['peak_location_ijk']}")
    print(f"    Peak cell centre (um):"
          f"({report['peak_location_um'][0]:.1f}, {report['peak_location_um'][1]:.1f}, "
          f"{report['peak_location_um'][2]:.1f})")
    print(f"    Location resolution:  one cell, {config.grid.dx * 1e6:.1f} x "
          f"{config.grid.dy * 1e6:.1f} x {config.grid.dz * 1e6:.1f} um")
    print(f"    Voxels > {threshold:.1f}°C:      {report['hotspot_count']}")
    p_w = config.power.die_power_w
    if p_w > 0:
        floor = regime_summary(config)["r_conv_k_per_w"]
        r_ja = (steady_peak - config.boundary_conditions.t_ambient_c) / p_w
        vs_floor = f", {r_ja / floor:.2f}x the 1/(hA) floor" if floor > 0 else ""
        print(f"    R_ja (measured):      {r_ja:.4f} K/W  [(peak - ambient) / P{vs_floor}]")
    print()

    # --- the verdict, judged on the steady state and only ever on that ---
    if steady_peak > threshold:
        print(f"  VERDICT: EXCEEDS THRESHOLD. Steady-state peak {steady_peak:.2f} °C is "
              f"above {threshold:.1f} °C")
        print(f"           in {report['hotspot_count']} voxel(s). This design needs "
              f"attention.")
    else:
        print(f"  VERDICT: within threshold. Steady-state peak {steady_peak:.2f} °C is "
              f"below {threshold:.1f} °C.")
    if config.case.expect_threshold_exceeded:
        if steady_peak > threshold:
            print("           (Expected: this configuration is a stress case -- see the banner.)")
        else:
            print(f"  NOTE: case '{config.case.name}' is a stress case, EXPECTED to exceed the "
                  f"threshold and did not. The case or the model has changed; check both.")
    if has_transient and t_sim < STEADY_STATE_TAU_MULTIPLE * tau_s:
        print(f"           (The transient above reached only {frac*100:.4f}% of steady "
              f"state and was NOT")
        print(f"            used for this verdict -- it cannot support one. See the "
              f"steady-state row.)")
    print()
    print(validation_text_block())
    print()
    print(f"  Interactive 3D view:   {report['html_path']}")
    print("=" * 60)


def stage2_catalog(config: PipelineConfig, scale_power_to_stage1: bool = True):
    """(catalog, note): the chiplet catalog Stage 2 will place, and one line
    saying how it was obtained. Stage 1 and Stage 2 must model ONE chip:

      - totals already equal: the catalog as authored;
      - rescaling allowed: every block scaled by one common factor to the
        Stage 1 die power (ratios preserved), and the note says so;
      - rescaling refused and the totals differ: PipelineError.

    The last case used to print a NOTE and run anyway -- a 36 W catalog under
    a 12 W Stage 1, two different chips in one invocation whose results
    cannot be compared. Shared by the startup banner, the pre-solve check in
    main() and run_stage2_optimizer(), so all three apply the same rule.
    """
    catalog = DEFAULT_CHIPLET_CATALOG
    authored_w = sum(c.power_w for c in catalog)
    target_w = config.power.die_power_w
    if math.isclose(authored_w, target_w, rel_tol=1e-12, abs_tol=1e-12):
        return catalog, (f"Catalog used as authored: {authored_w:.2f} W, equal to the "
                         f"Stage 1 die power.")
    if scale_power_to_stage1:
        return (scale_catalog_to_total_power(catalog, target_w),
                f"Catalog rescaled {authored_w:.2f} W -> {target_w:.2f} W to match the "
                f"Stage 1 die power (ratios preserved).")
    raise PipelineError(
        f"The Stage 2 catalog totals {authored_w:.2f} W but the Stage 1 die power is "
        f"{target_w:.2f} W, and --no-scale-catalog-power forbids rescaling: these describe "
        f"different chips. Set --die-power-w {authored_w:g} to run the catalog as authored, "
        f"or drop --no-scale-catalog-power to scale it to {target_w:g} W.")


def print_startup_banner(config: PipelineConfig, source: str,
                         duration_s: Optional[float] = None,
                         total_steps: Optional[int] = None,
                         optimize: bool = False,
                         scale_power_to_stage1: bool = True,
                         from_file: bool = False,
                         experiment: Optional[Experiment] = None,
                         output_dir_note: str = "") -> None:
    """Everything a result depends on, printed BEFORE any solve: where the
    configuration came from, the physical domain and its cells, the authored
    and realized stack with cell counts, the boundary conditions, the
    cooling regime, the power each stage uses, and the validation status.

    Units: SI inside the model (m, W, K, W/mK, W/m^2K). Micrometres,
    millimetres and degC appear only here and in the prompts.
    """
    g, s, bcs = config.grid, config.stack, config.boundary_conditions
    cells = s.layer_cells(g.nz, g.dz)       # validate() has already enforced this
    dz_um = g.dz * 1e6
    n = g.nx * g.ny * g.nz
    backend = "direct LU" if n <= DIRECT_SOLVE_MAX_UNKNOWNS else "conjugate gradients"

    print("=" * 60)
    print("RISC3D-AI :: 3D IC Thermal Simulation Pipeline")
    print("=" * 60)
    print(f"Config source: {source}")
    if from_file or config.case.name != "unnamed":
        print(f"Case: {config.case.name} -- {config.case.purpose}")
    if config.case.expect_threshold_exceeded:
        print(f"STRESS CASE: EXPECTED to exceed the "
              f"{config.visualization.hotspot_threshold_c:.1f} C threshold -- an EXCEEDS "
              f"verdict is the intended result, not a regression.")
    if output_dir_note:
        print(output_dir_note)
    print("Units: SI in the model (m, W, K); um, mm and degC appear only in this banner.")
    print(f"Domain: {g.nx * g.dx * 1e3:.3f} x {g.ny * g.dy * 1e3:.3f} mm x "
          f"{g.nz * dz_um:.3f} um = {g.nx} x {g.ny} x {g.nz} cells of "
          f"{g.dx * 1e6:.3f} x {g.dy * 1e6:.3f} x {dz_um:.3f} um")
    print(f"Unknowns: N = {n:,} ({backend}; direct limit {DIRECT_SOLVE_MAX_UNKNOWNS:,})")
    print("Stack (authored -> realized), bottom to top:")
    for name, t_um, k in zip(("interposer", "composite", "die"),
                             (s.interposer_um, s.bump_um, s.die_um), cells):
        print(f"  {name:<10}  {t_um:7.3f} um -> {k * dz_um:7.3f} um  ({k} cells)")
    print(f"  {'total':<10}  {s.total_um:7.3f} um = nz*dz = {g.nz} x {dz_um:.3f} um")
    # The die is DERIVED from the mesh (nx*dx by ny*dy), never stored, so it
    # cannot drift from it.
    die_w_mm, die_h_mm = g.nx * g.dx * 1e3, g.ny * g.dy * 1e3
    if config.chiplets:
        print(f"Power model:    {len(config.chiplets)} chiplets from the file; each "
              f"chiplet's power over its own footprint, through the die layer "
              f"z[{cells[0] + cells[1]}:{g.nz}]")
        catalog_mm2 = sum(c.width_um * c.height_um for c in config.chiplets) * 1e-6
        blocks = len(config.chiplets)
        largest = max(config.chiplets, key=lambda c: c.width_um * c.height_um)
        largest_mm = (largest.width_um / 1e3, largest.height_um / 1e3)
        fits = experiment is not None and not experiment.violations
    else:
        print(f"Power model:    one uniform footprint over the central "
              f"{config.power.die_footprint_frac:.0%} of the die; no chiplets defined, so "
              f"Stage 2 (if run) starts from the optimizer's own shelf-packed seed rather "
              f"than a placement you chose")
        catalog = DEFAULT_CHIPLET_CATALOG
        catalog_mm2 = sum(c.area_mm2 for c in catalog)
        blocks = len(catalog)
        largest = max(catalog, key=lambda c: c.area_um2)
        largest_mm = (largest.width_um / 1e3, largest.depth_um / 1e3)
        fits = preflight_optimizer_fit(config, catalog) is None
    print(f"Die:            {die_w_mm:.3f} x {die_h_mm:.3f} mm")
    print(f"Catalog:        {catalog_mm2:.2f} mm² across {blocks} blocks "
          f"({100.0 * catalog_mm2 / (die_w_mm * die_h_mm):.1f}% fill)")
    print(f"Catalog fits:   {'yes' if fits else 'no'} (largest block "
          f"{largest_mm[0]:.3f} x {largest_mm[1]:.3f} mm)")
    if config.chiplets:
        print(f"  {'chiplet':<8}{'name':<18}{'size mm':>14}{'power W':>10}"
              f"{'origin mm':>16}{'W/cm^2':>10}")
        for c in config.chiplets:
            x_um, y_um = config.initial_placement.origins_um[c.id]
            size = f"{c.width_um / 1e3:.2f} x {c.height_um / 1e3:.2f}"
            origin = f"({x_um / 1e3:.2f}, {y_um / 1e3:.2f})"
            density = c.power_w / (c.width_um * c.height_um * 1e-8)
            print(f"  {c.id:<8}{c.name:<18}{size:>14}{c.power_w:10.3f}"
                  f"{origin:>16}{density:10.2f}")
        print(f"Placement lattice: {config.initial_placement.pitch_um:.0f} um")
        if experiment is None:
            print("Initial placement: not validated")
        elif experiment.violations:
            print(f"Initial placement: INVALID -- {len(experiment.violations)} violation(s)")
            for violation in experiment.violations:
                print(f"  - {violation}")
        else:
            print(f"Initial placement: valid; {len(EXPERIMENT_CONSTRAINTS)} constraints "
                  f"checked (inside the package, no overlap, on the lattice, whole cells, "
                  f"fixed footprints and powers)")
    print(f"Ambient: {bcs.t_ambient_c:.1f}C | h_convective: {bcs.h_convective:.1f} W/m^2K | "
          f"Die power: {config.power.die_power_w:.2f}W")
    sink_face = "bottom" if bcs.bottom_bc != "neumann" else "top"
    sink_kind = bcs.bottom_bc if sink_face == "bottom" else bcs.top_bc
    face_desc = "die backside" if sink_face == "top" else "interposer underside"
    print(f"Sink: {sink_face} face ({face_desc}), {sink_kind}; all other faces adiabatic")
    print_regime_header(config)
    if optimize:
        try:
            _, power_note = stage2_catalog(config, scale_power_to_stage1)
            print(f"Stage 2 power: {power_note}")
        except PipelineError as exc:
            print(f"Stage 2 power: REFUSED -- {exc}")
    if duration_s is not None:
        print(f"Mode: steady state + transient {duration_s * 1e3:.4f} ms")
    elif total_steps is not None:
        print(f"Mode: steady state + transient ({total_steps:,} explicit steps)")
    else:
        print("Mode: steady state only (pass --transient <ms> to add a time series)")
    print(f"Status: {VALIDATION_STATUS.upper()} (limitations are listed with the results)")
    if from_file:
        # Accepted and validated, but nothing in this pipeline reads them.
        # Said here so a file's steady_state=false or method="crank_nicolson"
        # cannot look as if it took effect.
        print("Read from the file but not used by this pipeline: solver.method, "
              "solver.steady_state, solver.total_time_s, solver.dt, solver.record_steps, "
              "solver.max_steps, visualization.use_pyvista. The steady state is always "
              "solved; a transient runs only with --transient.")
    print("=" * 60)


def run_stage2_optimizer(
    config: PipelineConfig,
    open_browser: bool,
    iterations: int,
    seed: int,
    method: str = "exhaustive",
    placement_pitch_um: float = 2000.0,
    scale_power_to_stage1: bool = True,
    stage1_solver: Optional[ThermalSolver] = None,
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

    # One chip in both stages, or a refusal -- see stage2_catalog(). The
    # per-block wattage actually used is printed below either way.
    catalog, power_note = stage2_catalog(config, scale_power_to_stage1)
    print(f"[main] {power_note}")

    used_total = sum(c.power_w for c in catalog)
    print(f"[main] Catalog: {len(catalog)} chiplets totalling {used_total:.2f} W")
    for c in catalog:
        print(f"[main]   {c.name:<18} {c.power_w:6.2f} W  "
              f"{c.width_um:6.0f} x {c.depth_um:<6.0f} um  "
              f"{c.power_density_w_cm2:7.2f} W/cm^2")
    if method == "anneal":
        print(f"[main] Objective: steady-state peak temperature "
              f"({iterations} annealing proposals, seed={seed})")
    else:
        # Iterations and seed are annealer parameters; printing them on an
        # exhaustive run implied a sampled search that did not happen.
        print("[main] Objective: steady-state peak temperature "
              "(exhaustive: every buildable layout on the lattice; no seed)")

    try:
        optimizer = ChipletPlacementOptimizer.from_pipeline_config(
            config, catalog=catalog, seed=seed, reuse_solver=stage1_solver,
        )

        if method == "exhaustive":
            # Evaluates EVERY buildable layout on the placement lattice and
            # returns the proven global optimum. Affordable because the
            # steady-state system is linear: one solve per (chiplet, position)
            # pair, after which each candidate costs an array sum rather than
            # a linear solve. See src/exhaustive.py.
            search = ExhaustivePlacementSearch(
                optimizer, placement_pitch_um=placement_pitch_um,
            )
            exhaustive_result = search.search(verbose=True)
            print("\n" + exhaustive_result.summarize())
            result = exhaustive_result.to_optimization_result(optimizer)
        else:
            exhaustive_result = None
            result = optimizer.optimize(iterations=iterations, verbose=True)
            print("\n" + result.summarize())
            print("  NOTE: simulated annealing SAMPLES the placement space. This is the")
            print("        best of the layouts it happened to visit, NOT a proven optimum,")
            print("        and a different --optimize-seed gives a different answer. Use")
            print("        --optimize-method exhaustive for a guaranteed global best.")

        report_path = render_optimization_report(
            result, output_dir=config.visualization.output_dir, method=method,
        )
    except (OptimizerError, ExhaustiveSearchError) as exc:
        raise PipelineError(f"Stage 2 placement optimization failed: {exc}") from exc

    print(validation_text_block())
    print(f"  Optimization report:   {report_path}")

    _open_in_browser(report_path, open_browser, "the optimization report")
    return {"result": result, "exhaustive": exhaustive_result, "report_path": report_path}


def print_experiment_comparison(experiment: Experiment) -> None:
    """The before/after block: what the optimizer was asked, what it moved,
    and how the real solver's re-solve compares with what it reported."""
    record = experiment.optimization
    initial, optimized = experiment.initial, experiment.optimized
    improvement = experiment.improvement()
    print("\n" + "=" * 60)
    print("BEFORE / AFTER :: PLACEMENT OPTIMIZATION")
    print("=" * 60)
    print(f"  Objective:              {record.objective}")
    reproducible = "deterministic" if record.deterministic else f"seed {record.seed}"
    print(f"  Method:                 {record.method} on a {record.pitch_m * 1e6:.0f} um "
          f"lattice ({reproducible})")
    total = f" of {record.candidates_total:,}" if record.candidates_total else ""
    print(f"  Candidates evaluated:   {record.candidates_evaluated:,}{total} "
          f"({record.rejected_or_pruned:,} rejected or pruned)")
    if record.superposition_error_k is not None:
        print(f"  Superposition check:    {record.superposition_error_k:.2e} K against a "
              f"direct solve")
    print(f"  Optimizer runtime:      {record.runtime_s:.2f} s")
    print(f"  Initial peak:           {initial.peak_c:.4f} degC (rise {initial.rise_k:.4f} K)")
    print(f"  Optimized peak:         {optimized.peak_c:.4f} degC "
          f"(rise {optimized.rise_k:.4f} K)")
    print(f"  Improvement:            {improvement['delta_peak_k']:.4f} K cooler "
          f"({improvement['pct_of_initial_rise']:.1f}% of the initial rise; "
          f"{improvement['pct_of_initial_excess']:.1f}% of the part above the P/(hA) floor)")
    print(f"  Re-solve check:         optimizer reported {record.optimizer_peak_c:.4f} degC, "
          f"the real solver returns {optimized.peak_c:.4f} degC (delta "
          f"{record.resolve_delta_k:.1e} K, field {record.resolve_field_delta_k:.1e} K)")
    print("  Moves:")
    print(f"    {'id':<8}{'initial mm':>16}{'optimized mm':>16}{'moved mm':>10}"
          f"   peak before -> after")
    for row in experiment.moves():
        start = f"({row.x0_m * 1e3:.1f}, {row.y0_m * 1e3:.1f})"
        end = f"({row.x1_m * 1e3:.1f}, {row.y1_m * 1e3:.1f})"
        print(f"    {row.id:<8}{start:>16}{end:>16}{row.distance_m * 1e3:10.2f}"
              f"   {row.peak0_c:6.2f} -> {row.peak1_c:6.2f} degC")
    print("=" * 60)


def run_experiment_pipeline(experiment: Experiment, *, open_browser: bool,
                            optimize: bool, method: str, iterations: int,
                            seed: int) -> dict:
    """The product workflow: a validated initial placement, its baseline
    solve, an optional optimization from it, a real re-solve of the winner,
    and the before/after report."""
    timing = TimingReport()
    config = experiment.config

    initial = experiment.solve_initial()
    timing.grid_build_s = experiment.build_s
    timing.solver_init_s = experiment.solver_init_s
    timing.steady_state_s = initial.solve_s
    tau_s = experiment.solver.estimate_time_constant()
    print(f"[main] Package thermal time constant tau = {tau_s * 1e3:.2f} ms "
          f"(C/G from the actual grid).")
    print(f"[main] Initial placement solved: peak {initial.peak_c:.2f} °C "
          f"(equilibrium, t -> infinity) in {initial.solve_s:.4f}s")
    print(f"[main] Power deposited: {initial.power.deposited_w:.6f} W over "
          f"{initial.power.powered_cells:,} cells; heat leaving the sink face "
          f"{initial.sink_heat_w:.6f} W")

    t0 = time.perf_counter()
    try:
        viz = ThermalVisualizer(
            temperature_k=np.asarray(initial.field_k), dx=config.grid.dx,
            dy=config.grid.dy, dz=config.grid.dz,
            output_dir=config.visualization.output_dir)
        initial_html = viz.render_interactive_webgl(
            filename="thermal_3d.html",
            title="RISC3D-AI :: Initial Placement, Steady State",
            banner_html=('<div class="banner"><b>Initial placement</b>, steady state '
                         '(t &rarr; &infin;). This is the baseline the optimization is '
                         'measured against.</div>'
                         f'<div class="banner">{validation_html_block()}</div>'))
        hotspots = viz.get_hotspot_report(
            threshold_c=config.visualization.hotspot_threshold_c)
    except VisualizationError as exc:
        raise PipelineError(f"WebGL visualization export failed: {exc}") from exc
    _open_in_browser(initial_html, open_browser, "the 3D thermal view")
    timing.visualization_s = time.perf_counter() - t0

    print_stage1_results(config, {
        "html_path": initial_html,
        "hotspot_count": hotspots.count,
        "peak_temperature_c": initial.peak_c,
        "peak_location_um": tuple(v * 1e6 for v in initial.peak_center_m),
        "peak_location_ijk": initial.peak_ijk,
        "transient_peak_temperature_c": None,
        "transient_duration_s": None,
        "transient_html_path": None,
        "tau_s": tau_s,
        "converged_fraction": 0.0,
    })

    optimized_html = None
    if optimize:
        t0 = time.perf_counter()
        experiment.optimize(method=method, iterations=iterations, seed=seed, verbose=True)
        timing.optimize_s = time.perf_counter() - t0
        print_experiment_comparison(experiment)
        t0 = time.perf_counter()
        try:
            optimized_viz = ThermalVisualizer(
                temperature_k=np.asarray(experiment.optimized.field_k),
                dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
                output_dir=config.visualization.output_dir)
            optimized_html = optimized_viz.render_interactive_webgl(
                filename="thermal_3d_optimized.html",
                title="RISC3D-AI :: Optimized Placement, Steady State",
                banner_html=('<div class="banner"><b>Optimized placement</b>, steady state, '
                             're-solved by the real solver after the search.</div>'
                             f'<div class="banner">{validation_html_block()}</div>'))
        except VisualizationError as exc:
            raise PipelineError(f"WebGL visualization export failed: {exc}") from exc
        _open_in_browser(optimized_html, open_browser, "the optimized 3D view")
        timing.visualization_s += time.perf_counter() - t0

    report_path = render_experiment_report(
        experiment, output_dir=config.visualization.output_dir,
        links={"initial_3d": initial_html, "optimized_3d": optimized_html})
    print(f"  Experiment report:     {report_path}")
    _open_in_browser(report_path, open_browser, "the experiment report")
    return {"timing": timing, "experiment": experiment, "report_path": report_path}


def run_pipeline(
    config: PipelineConfig,
    total_steps: Optional[int],
    duration_s: Optional[float],
    open_browser: bool,
    run_optimizer: bool = False,
    prompt_for_optimizer: bool = False,
    optimize_iterations: int = 30,
    optimize_seed: int = 0,
    optimize_method: str = "exhaustive",
    placement_pitch_um: float = 2000.0,
    scale_power_to_stage1: bool = True,
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

    # --- Phase 3: time constant, and the CFL bound only if time-stepping ---
    # Steady state is the DEFAULT and needs no timestep: it is one linear
    # solve. Time-stepping happens only when a transient was explicitly
    # requested. The old default spent 11,117 steps and 7.1 s reaching 4.88%
    # of the steady rise, then printed a verdict saying that run had not been
    # used for anything -- pure cost for a discarded number.
    #
    # NOTE: reads ONLY SolverSection.safety_factor (via getattr with a
    # hardcoded fallback for extra defensiveness against a stale/older
    # local config_loader.py). Never reads .method or .dt.
    safety_factor = getattr(config.solver, "safety_factor", 0.8)
    tau_s = solver.estimate_time_constant()
    print(f"[main] Package thermal time constant tau = {tau_s*1e3:.2f} ms "
          f"(C/G from the actual grid).")

    want_transient = (duration_s is not None) or (total_steps is not None)
    dt_stable = None
    total_time_s = 0.0
    converged_fraction = 0.0

    if want_transient:
        t0 = time.perf_counter()
        try:
            dt_stable = solver.compute_max_dt(safety_factor=safety_factor)
        except ThermalSolverError as exc:
            raise PipelineError(f"CFL stability check failed: {exc}") from exc
        timing.cfl_check_s = time.perf_counter() - t0

        if duration_s is not None:
            total_time_s = float(duration_s)
            total_steps = max(1, int(np.ceil(total_time_s / dt_stable)))
            print(f"[main] --transient {total_time_s*1e3:.4f} ms / dt_stable "
                  f"{dt_stable:.6e}s -> {total_steps:,} explicit steps")
        else:
            total_time_s = dt_stable * total_steps
            print(f"[main] --total-steps {total_steps:,} at dt_stable {dt_stable:.6e}s "
                  f"-> {total_time_s*1e3:.4f} ms")

        converged_fraction = solver.steady_state_fraction(total_time_s)
        print(f"[main] Transient span {total_time_s*1e3:.4f} ms = "
              f"{total_time_s/tau_s:.6f} tau, reaching ~{converged_fraction*100:.4f}% of "
              f"the steady-state rise.")
        if total_time_s < STEADY_STATE_TAU_MULTIPLE * tau_s:
            steps_for_3tau = int(np.ceil(STEADY_STATE_TAU_MULTIPLE * tau_s / dt_stable))
            print(f"[main] NOTE: a TRANSIENT snapshot, not an equilibrium. Reaching "
                  f"{STEADY_STATE_TAU_MULTIPLE:.0f}*tau would need {steps_for_3tau:,} steps. "
                  f"Threshold checks use the steady state below, never this.")
    else:
        print("[main] Steady-state mode (default): no time-stepping, no CFL bound needed. "
              "Pass --transient <ms> for a time series.")

    # --- Phase 4: Transient solve, only when asked for ---
    result = None
    transient_series = None
    if want_transient:
        # Log-spaced samples across the span, so a decade-wide transient is
        # resolved at every scale rather than only near the end. A single
        # end-of-run instant hides the whole trajectory.
        n_samples = 40
        sample_times = np.unique(np.geomspace(
            max(dt_stable, total_time_s / 1e4), total_time_s, n_samples
        ))
        t0 = time.perf_counter()
        try:
            solver.reset()
            series_t: List[float] = []
            series_peak: List[float] = []
            elapsed = 0.0
            for target in sample_times:
                span = float(target) - elapsed
                if span <= 0:
                    continue
                solver.run(total_time_s=span, dt=dt_stable,
                           safety_factor=safety_factor, record_interval_s=None,
                           max_steps=int(np.ceil(span / dt_stable)) + 10)
                elapsed = solver.time_s
                series_t.append(elapsed)
                series_peak.append(float(np.max(solver.T)) - 273.15)
            result = {
                "final_T": solver.T.copy(),
                "final_time_s": solver.time_s,
                "steps_taken": solver.step_count,
            }
            transient_series = {"t_s": series_t, "peak_c": series_peak}
        except ThermalSolverError as exc:
            raise PipelineError(f"Transient solve failed: {exc}") from exc
        timing.solve_s = time.perf_counter() - t0
        print(f"[main] Transient complete: {result['steps_taken']:,} steps, "
              f"final_time={result['final_time_s']:.6e}s, "
              f"{len(series_t)} samples recorded")

    # --- Phase 4b: Steady state, ALWAYS ---
    # The transient above answers "how hot is it at t = <duration>". The
    # threshold question -- "is this design safe" -- is about equilibrium, and
    # a run of a fraction of a tau cannot answer it. This is one cached linear
    # solve, so there is no reason not to have the real answer.
    t0 = time.perf_counter()
    try:
        steady_T = solver.solve_steady_state(apply_result=False)
    except ThermalSolverError as exc:
        raise PipelineError(f"Steady-state solve failed: {exc}") from exc
    timing.steady_state_s = time.perf_counter() - t0
    steady_peak_c = float(np.max(steady_T)) - 273.15
    print(f"[main] Steady-state solve: peak {steady_peak_c:.2f} °C "
          f"(equilibrium, t -> infinity) in {timing.steady_state_s:.4f}s")

    # --- Phase 5: Interactive WebGL 3D export (replaces static PNG output) ---
    t0 = time.perf_counter()
    try:
        # thermal_3d.html RENDERS THE STEADY STATE. It used to render the
        # transient, so the browser showed a 30.93 C field while the terminal
        # reported a 79.97 C verdict -- the default-opened artifact disagreeing
        # with the printed conclusion. The picture and the verdict now describe
        # the same field by construction.
        steady_viz = ThermalVisualizer(
            temperature_k=steady_T,
            dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
            output_dir=config.visualization.output_dir,
        )
        html_path = steady_viz.render_interactive_webgl(
            filename="thermal_3d.html",
            title="RISC3D-AI :: Steady-State Thermal Field",
            banner_html=(
                '<div class="banner"><b>Steady state</b> (t &rarr; &infin;). This is the '
                'equilibrium field the threshold verdict is judged against.</div>'
                f'<div class="banner">{validation_html_block()}</div>'
            ),
        )
        hotspot_report = steady_viz.get_hotspot_report(
            threshold_c=config.visualization.hotspot_threshold_c
        )

        # A transient, when one was requested, is written SEPARATELY with its
        # duration in the filename and an on-page banner giving t, tau and the
        # fraction of steady rise reached. It can then never be mistaken for
        # the equilibrium result.
        transient_html_path = None
        transient_peak = None
        if result is not None:
            transient_viz = ThermalVisualizer(
                temperature_k=result["final_T"],
                dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
                output_dir=config.visualization.output_dir,
            )
            transient_peak = transient_viz.get_peak_hotspot()
            t_ms = result["final_time_s"] * 1e3
            transient_html_path = transient_viz.render_interactive_webgl(
                filename=f"thermal_3d_transient_{t_ms:.4f}ms.html",
                title=f"RISC3D-AI :: Transient Field at t = {t_ms:.4f} ms",
                banner_html=(
                    f'<div class="banner"><b>TRANSIENT SNAPSHOT &mdash; not equilibrium.</b><br>'
                    f't = {t_ms:.4f} ms = {result["final_time_s"]/tau_s:.6f} &tau; '
                    f'(&tau; = {tau_s*1e3:.2f} ms), which is '
                    f'{converged_fraction*100:.4f}% of the steady-state rise.<br>'
                    f'The steady state is in <b>thermal_3d.html</b>, and every threshold '
                    f'verdict uses that field, not this one.</div>'
                    f'<div class="banner">{validation_html_block(transient=True)}</div>'
                ),
            )
            print(f"[main] Transient snapshot written separately: {transient_html_path}")

        _open_in_browser(html_path, open_browser, "the 3D thermal view")

        report = {
            "html_path": html_path,
            # Hotspot count / peak / location all describe the STEADY state,
            # which is what the threshold verdict is about.
            "hotspot_count": hotspot_report.count,
            "peak_temperature_c": hotspot_report.peak_temperature_c,
            "peak_location_um": hotspot_report.peak_location_um,
            "peak_location_ijk": hotspot_report.peak_location_ijk,
            # The transient snapshot, kept separate and always labelled with
            # the duration it corresponds to.
            "transient_peak_temperature_c": (
                transient_peak["temperature_c"] if transient_peak is not None else None),
            "transient_duration_s": (
                result["final_time_s"] if result is not None else None),
            "transient_html_path": transient_html_path,
            "transient_series": transient_series,
            "tau_s": tau_s,
            "converged_fraction": converged_fraction,
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
            method=optimize_method, placement_pitch_um=placement_pitch_um,
            scale_power_to_stage1=scale_power_to_stage1,
            stage1_solver=solver,
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

    given_flags = _explicit_flags(sys.argv[1:] if argv is None else argv)
    output_dir_note = ""

    try:
        if args.config is not None:
            clashing = sorted(given_flags & CONFIG_FILE_CONFLICTS)
            if clashing:
                raise ConfigError(
                    f"{', '.join(clashing)} cannot be combined with --config: the file is "
                    f"the single source of truth for what is simulated, and a flag beside "
                    f"it was previously ignored in silence. Edit the file, or drop --config.")
            config = load_config(args.config)
            if config.chiplets and "--placement-pitch-um" in given_flags:
                raise ConfigError(
                    "--placement-pitch-um cannot be combined with a config file that "
                    "defines initial_placement.pitch_um: the file already states the "
                    "lattice the initial placement sits on.")
            if "--output-dir" in given_flags:
                output_dir_note = (
                    f"Output directory: {args.output_dir} (from --output-dir, overriding "
                    f"the file's '{config.visualization.output_dir}')")
                config.visualization.output_dir = args.output_dir
            total_steps = args.total_steps
            duration_s = args.transient * 1e-3 if args.transient is not None else None
        elif argv is None and len(sys.argv) == 1:
            # `python main.py` with truly no arguments -> interactive prompts.
            config, total_steps, duration_s = run_interactive_prompts()
            interactive = True
        else:
            # Any explicit flag supplied -> non-interactive, flag-driven run.
            config, total_steps, duration_s = build_config_from_flags(args)
            if args.transient is not None:
                duration_s = args.transient * 1e-3
    except ConfigError as exc:
        print(f"\n[FATAL] Configuration error: {exc}", file=sys.stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\n[main] Input cancelled. Exiting.")
        return 1

    source = (f"config file {args.config}" if args.config is not None
              else "interactive prompts" if interactive else "command-line flags")
    # A configuration with chiplets is a placement EXPERIMENT. Built before
    # the banner so the banner can show the initial placement and whether it
    # is valid, and so an invalid one is refused before anything is solved.
    experiment = None
    if config.chiplets:
        try:
            experiment = Experiment(config, source=source)
        except (ExperimentError, ConfigError) as exc:
            print(f"\n[FATAL] {exc}", file=sys.stderr)
            return EXIT_PREFLIGHT_FAILED

    print_startup_banner(
        config, source, duration_s=duration_s, total_steps=total_steps,
        optimize=args.optimize, scale_power_to_stage1=not args.no_scale_catalog_power,
        from_file=args.config is not None, experiment=experiment,
        output_dir_note=output_dir_note,
    )

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
    if experiment is not None and experiment.violations:
        print("\n[FATAL] The initial placement is not physically valid, so nothing was "
              "solved:", file=sys.stderr)
        for violation in experiment.violations:
            print(f"  - {violation}", file=sys.stderr)
        return EXIT_PREFLIGHT_FAILED

    if experiment is not None and (duration_s is not None or total_steps is not None):
        print("\n[FATAL] --transient / --total-steps are not supported for a placement "
              "experiment: a before/after comparison is a steady-state result. Run a "
              "transient from a configuration without chiplets.", file=sys.stderr)
        return 1

    if args.optimize and experiment is None:
        preflight_error = preflight_optimizer_fit(config)
        if preflight_error is None:
            # Stage 1 and Stage 2 must describe one chip. Checked here, before
            # any solve, so a refusal costs nothing.
            try:
                stage2_catalog(config, not args.no_scale_catalog_power)
            except PipelineError as exc:
                preflight_error = str(exc)
        if preflight_error is not None:
            print(f"\n[FATAL] {preflight_error}", file=sys.stderr)
            return EXIT_PREFLIGHT_FAILED

    try:
        if experiment is not None:
            outcome = run_experiment_pipeline(
                experiment, open_browser=not args.no_browser, optimize=args.optimize,
                method=args.optimize_method, iterations=args.optimize_iterations,
                seed=args.optimize_seed)
        else:
            outcome = run_pipeline(
                config, total_steps, duration_s,
                open_browser=not args.no_browser,
                run_optimizer=args.optimize,
                prompt_for_optimizer=interactive,
                optimize_iterations=args.optimize_iterations,
                optimize_seed=args.optimize_seed,
                optimize_method=args.optimize_method,
                placement_pitch_um=args.placement_pitch_um,
                scale_power_to_stage1=not args.no_scale_catalog_power,
            )
    except (PipelineError, ExperimentError) as exc:
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