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
# Measured on the default 40x40x12 grid, 4-block catalog scaled to P,
# exhaustive search on the 2000 um lattice:
#
#   preset            P      h       lambda   Stage 2: initial -> optimum    gain
#   passive-lowpower  12 W   1000    5.47 mm   83.30 ->  70.24 degC   13.06 K (22.4% of rise)
#   cowos-coldplate   150 W  25000   1.09 mm  105.45 -> 100.82 degC    4.63 K ( 5.8% of rise)
#
# With lambda comparable to the block spacing, blocks share heat and moving
# them apart pays. With lambda well under it, each block sees essentially
# only its own footprint -- the 150 W Stage 1 rise of 59.11 K is within 1.5%
# of the local 1-D q''/h = 60 K -- and placement is a weak lever. Placement
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

# Materials of the stack build_chiplet_stack() assembles, bottom to top.
# Shared with the spreading-length calculation so the two cannot drift apart
# when a material's properties change.
STACK_MATERIALS = ("silicon", "bump_underfill_composite", "silicon")


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
    print(f"Heat flux: {r['flux_footprint_w_cm2']:.2f} W/cm^2 over the active footprint "
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
    The heat sink is the TOP face (z = nz-1, the die backside), which is how a
    CoWoS part actually cools: die -> TIM1 -> lid -> TIM2 -> heatsink. The
    bottom face (interposer underside) and all four sides are adiabatic.
    Earlier versions sank through the bottom with an adiabatic top -- the
    package upside down. Which face is the sink comes from
    config.boundary_conditions.bottom_bc / top_bc, not from this function.
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
        gb.assign_layer(slice(0, z_interposer_end), STACK_MATERIALS[0])
        gb.assign_layer(slice(z_interposer_end, z_bump_end), STACK_MATERIALS[1])
        gb.assign_layer(slice(z_bump_end, nz), STACK_MATERIALS[2])
        gb.tag_boundaries()

        x_slice, y_slice = active_footprint_slices(config)
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
        f"over footprint x[{x_slice.start}:{x_slice.stop}] y[{y_slice.start}:{y_slice.stop}]"
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
    print(f"    Peak location (um):   "
          f"({report['peak_location_um'][0]:.1f}, {report['peak_location_um'][1]:.1f}, "
          f"{report['peak_location_um'][2]:.1f})")
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
    if has_transient and t_sim < STEADY_STATE_TAU_MULTIPLE * tau_s:
        print(f"           (The transient above reached only {frac*100:.4f}% of steady "
              f"state and was NOT")
        print(f"            used for this verdict -- it cannot support one. See the "
              f"steady-state row.)")
    print()
    print(f"  Interactive 3D view:   {report['html_path']}")
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

    # Stage 1 asked the user for a die power. Stage 2 used to ignore it and
    # run DEFAULT_CHIPLET_CATALOG's 36.0 W regardless -- a 3x mismatch at the
    # 12 W default, which is two different chips in one invocation with
    # energy-balance floors 60 K apart. Whatever is chosen here, the per-block
    # wattage actually used is printed.
    catalog = DEFAULT_CHIPLET_CATALOG
    catalog_total = sum(c.power_w for c in catalog)
    stage1_power = config.power.die_power_w
    if scale_power_to_stage1 and abs(catalog_total - stage1_power) > 1e-9:
        catalog = scale_catalog_to_total_power(catalog, stage1_power)
        print(f"[main] Catalog rescaled {catalog_total:.1f} W -> {stage1_power:.1f} W to match "
              f"the Stage 1 die power (ratios preserved).")
    elif abs(catalog_total - stage1_power) > 1e-9:
        print(f"[main] NOTE: catalog totals {catalog_total:.1f} W but Stage 1 was given "
              f"{stage1_power:.1f} W. Running the catalog as authored -- these are two "
              f"different chips, and the Stage 1 and Stage 2 numbers are not comparable.")

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
            result, output_dir=config.visualization.output_dir,
        )
    except (OptimizerError, ExhaustiveSearchError) as exc:
        raise PipelineError(f"Stage 2 placement optimization failed: {exc}") from exc

    print(f"  Optimization report:   {report_path}")

    _open_in_browser(report_path, open_browser, "the optimization report")
    return {"result": result, "exhaustive": exhaustive_result, "report_path": report_path}


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

    try:
        if args.config is not None:
            if args.preset is not None:
                raise ConfigError(
                    "--preset applies to flag-driven runs; a --config file sets its own "
                    "die power and h_convective. Pass one or the other.")
            config = load_config(args.config)
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

    print("=" * 60)
    print("RISC3D-AI :: 3D IC Thermal Simulation Pipeline")
    print("=" * 60)
    print(f"Grid: {config.grid.nx}x{config.grid.ny}x{config.grid.nz} voxels "
          f"({config.grid.dx*1e6:.1f}x{config.grid.dy*1e6:.1f}x{config.grid.dz*1e6:.1f} um)")
    print(f"Ambient: {config.boundary_conditions.t_ambient_c:.1f}C | "
          f"h_convective: {config.boundary_conditions.h_convective:.1f} W/m^2K | "
          f"Die power: {config.power.die_power_w:.2f}W")
    bcs = config.boundary_conditions
    sink_face = "bottom" if bcs.bottom_bc != "neumann" else "top"
    sink_kind = bcs.bottom_bc if sink_face == "bottom" else bcs.top_bc
    face_desc = "die backside" if sink_face == "top" else "interposer underside"
    print(f"Sink: {sink_face} face ({face_desc}), {sink_kind}; all other faces adiabatic")
    print_regime_header(config)
    if duration_s is not None:
        print(f"Mode: steady state + transient {duration_s*1e3:.4f} ms")
    elif total_steps is not None:
        print(f"Mode: steady state + transient ({total_steps:,} explicit steps)")
    else:
        print("Mode: steady state only (pass --transient <ms> to add a time series)")
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