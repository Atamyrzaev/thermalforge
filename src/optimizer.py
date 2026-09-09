"""
optimizer.py
RISC3D-AI :: Stage 2 -- Automated Chiplet Placement Optimizer

Stage 1 (main.py) answers "how hot does THIS stack get?". Stage 2 answers
"where should the chiplets go so it gets as cool as possible?".

Given a catalog of chiplets (each a rectangular XY footprint carrying a fixed
power budget) and a fixed vertical stack, ChipletPlacementOptimizer searches
the space of non-overlapping XY placements on the die layer with simulated
annealing, minimizing the steady-state peak junction temperature.

Physics / performance notes
---------------------------
- The objective is the STEADY-STATE peak temperature, obtained from
  ThermalSolver.solve_steady_state(). That is a single sparse linear solve,
  not a time integration, so the optimizer never needs a timestep, a CFL
  bound, or any transient parameter. Callers must not thread dt/total_steps
  in here -- there is nothing for them to do.

- Moving a chiplet changes ONLY the volumetric power density map q_map. It
  does not change the material maps (kx/ky/kz, rho, cp) or the boundary
  configuration, and therefore does not change the sparse conduction operator
  M or the boundary forcing vector f that ThermalSolver assembles in
  __init__. So a single ThermalSolver is built once and its q_map is mutated
  in place between candidates; solve_steady_state() re-reads q_map on every
  call (it recomputes `forcing` from self.q_map each time), which makes this
  reuse exact rather than an approximation. It also lets the solver cache its
  factorization across the whole anneal: below DIRECT_SOLVE_MAX_UNKNOWNS the
  LU of M is computed once and every candidate is then a triangular solve
  (measured 0.136 s per candidate against a 23 s factorization at N=38,400).

- ALWAYS sanity-check the reported peak against the convective floor rather
  than accepting it. R_conv ~ 1/(h*A), so the peak rise above ambient should
  land near P/(h*A); the measured excess over that floor is conduction
  spreading resistance. OptimizationResult exposes this directly as
  predicted_convective_rise_k and convective_floor_ratio, and summarize()
  prints both. A peak far from the floor is a defect, not merely a hot chip.
  Measured for the reference configuration: floor 90.0 K, measured rise
  141.4 K, ratio 1.5714 -- the excess is real spreading resistance at 48%
  die fill, where heat must travel laterally through a thin low-k stack to
  reach the whole boundary. The floor is a LOWER BOUND, not a prediction: it
  assumes heat leaves uniformly across the entire die area, which it does
  not.

  Note carefully that OptimizationResult.improvement_c is the DIFFERENCE
  between two peak temperatures, not a rise above ambient. It must never be
  fed into P*R_conv; doing so back-derives a die size that was never used.

- MESH CONVERGENCE. Once footprints and stack layers were both authored in
  micrometers, the geometry became identical at every refinement level and
  the peak turned out to be very nearly mesh-independent. Measured on the
  r=2 ladder over the same 20 x 20 mm die, all with exact 100/40/100 um
  layers:

      20x20x12 (N=4,800)     peak 200.2271 degC
      40x40x24 (N=38,400)    peak 200.3900 degC
      80x80x48 (N=307,200)   peak 200.4035 degC

  That is a 0.18 degC (0.09%) spread across a 64x range of unknowns, rising
  monotonically and converging from below as coarse meshes smear the hotspot.
  An earlier 143-166 degC spread was NOT discretization error: it came from
  fraction-authored stack layers silently changing the composite thickness
  between meshes (see config_loader.StackSection). A formal GCI is still
  outstanding -- the observed order p is around 3.6, above the nominal 2, so
  the triplet may not be in the asymptotic range and the GCI band would be
  optimistic. Quote the number with that caveat until the study is finished.

- SOLVER BACKEND. The steady-state solve used to fail outright on package-
  scale grids: above DIRECT_SOLVE_MAX_UNKNOWNS it fell to a Jacobi-
  preconditioned BiCGSTAB that did not converge on an anisotropic mesh. That
  is FIXED -- thermal_solver.solve_steady_state() now runs conjugate
  gradients on the exactly-symmetric SPD form of the operator, and every
  configuration in the ladder above solves cleanly, including 50:1 aspect
  ratios. See STEADY-STATE BACKEND in thermal_solver.py for the measurements.

  Large grids remain SLOWER (CG iteration count grows with N: 935 iterations
  at N=4,800, 3,062 at 38,400, 8,425 at 307,200) but slowness is a
  scheduling problem, not a correctness one, and every solve is
  residual-verified and energy-balance-verified before it is returned.

Invalid-candidate handling (the load-bearing invariant)
-------------------------------------------------------
A candidate layout whose chiplets overlap is not physically buildable, so it
is never solved -- there is no temperature field for it and its peak
temperature is None. It is still assigned a cost (a large overlap penalty)
so the annealer can rank and reject it.

That means cost and peak temperature can disagree about validity, and the
elitist best-tracker must key on BOTH. _consider_candidate() therefore
requires `cand_peak is not None` in addition to `cand_cost < best_cost`
before promoting a candidate to best. Dropping that guard lets an
unbuildable overlapping layout become the reported optimum with
best_peak_temperature_c left as None, which then propagates a None into
every downstream consumer (the report renderer, main.py's summary print).
See TestChipletPlacementOptimizer in tests/test_core.py, which pins this
invariant directly.

Report output
-------------
render_optimization_report() writes a single self-contained HTML file. It
follows this codebase's established house style for HTML generation --
a module-level template string with __PLACEHOLDER__ tokens substituted via
str.replace(), exactly as src/visualization.py does -- and embeds its two
figures as base64 PNGs rendered with the Matplotlib that visualization.py
already depends on. There is deliberately NO Plotly dependency: this project
does not have plotly installed and does not use it anywhere, and adding a
third-party plotting stack for one report would be a real dependency for a
cosmetic gain.

Reference configuration
-----------------------
The default catalog is 36.0 W of realistic advanced-packaging blocks over
192 mm^2, which needs a package-scale die rather than Stage 1's 1 mm^2
default grid. The configuration every measurement in this docstring was
taken from:

    python main.py --optimize --nx 40 --ny 40 --nz 24
                   --dx-um 500 --dy-um 500 --dz-um 10 --h-convective 1000
                   --optimize-iterations 30 --optimize-seed 7
    (one command; wrapped here for width)

That is a 20 x 20 mm die at 48% fill, with an exact 100/40/100 um stack
(nz=24 at dz=10 um = 240 um total). Measured result: initial peak
200.39 degC, optimized peak 166.43 degC, improvement 33.96 degC (16.95%)
over 30 annealing proposals, 76.6 s optimizer wall clock. All four catalog
footprints resolve to an exact whole number of voxels at a 500 um pitch
(0.0% area error), and all origins lie on the 1000 um placement lattice.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.config_loader import StackSection, ConfigError
from src.grid_builder import GridBuilder, GridBuilderError
from src.thermal_solver import (
    ThermalSolver, ThermalSolverError, DIRECT_SOLVE_MAX_UNKNOWNS,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.Optimizer")


__all__ = [
    "OptimizerError",
    "ChipletSpec",
    "OptimizationResult",
    "ChipletPlacementOptimizer",
    "DEFAULT_CHIPLET_CATALOG",
    "scale_catalog_to_total_power",
    "render_optimization_report",
    "OVERLAP_PENALTY_C",
    "OVERLAP_PENALTY_PER_CELL_C",
]


# Overlap penalty, expressed in the same units as the objective (degrees C).
# Deliberately far above any physically reachable peak temperature so that an
# unbuildable layout can never out-rank a buildable one on cost alone. The
# `cand_peak is not None` guard in _consider_candidate() is the actual
# correctness invariant; this large base is defense in depth, not a substitute.
OVERLAP_PENALTY_C = 1.0e6
OVERLAP_PENALTY_PER_CELL_C = 50.0

# Physical lattice, in micrometers, that chiplet ORIGINS are snapped to.
#
# Chiplet extents already divide exactly at every pitch in the refinement
# ladder, but that is not sufficient for a mesh-convergence study: the
# origins must also be representable on the COARSEST mesh, or the geometry
# differs between levels and the study is comparing different problems.
#
#     origin x = 3500 um -> exact at dx = 1000/500/250 um?  [False, True, True]
#     origin x = 3000 um -> exact at dx = 1000/500/250 um?  [True,  True, True]
#
# 1000 um is the coarsest pitch in the approved ladder (20x20 over a 20 mm
# die), so snapping origins to it makes every layout representable at every
# level. It is also a realistic floorplanning constraint in its own right --
# real placement grids are coarse -- so this costs nothing physically.
#
# This is the same failure mode as the chiplet-footprint and stack-layer bugs,
# one level up: geometry defined in grid units rather than physical ones.
DEFAULT_PLACEMENT_PITCH_UM = 1000.0


class OptimizerError(Exception):
    """Raised on an invalid catalog, an infeasible die footprint, or a failed
    candidate evaluation."""


# ---------------------------------------------------------------------------
# Chiplet catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipletSpec:
    """A single placeable chiplet: a rectangular XY footprint in MICROMETERS
    carrying a fixed total power budget in watts. The footprint spans the
    full die z-range; only its XY origin is a free variable.

    Dimensions are physical, never voxel counts. A chiplet is a piece of
    silicon with a real size, and that size cannot be a function of how
    finely the solver happens to mesh the domain. Authoring in voxels would
    make a convergence study meaningless: halving dx would halve every
    chiplet's linear extent and quadruple its areal power density, so
    refining the mesh would silently change the physics being modelled --
    exactly backwards. extent_vox() resolves a physical size against a
    specific dx/dy at build time, which is the single point where mesh
    resolution is allowed to enter.
    """
    name: str
    width_um: float     # X extent, micrometers
    depth_um: float     # Y extent, micrometers
    power_w: float

    def validate(self) -> None:
        if self.width_um <= 0 or self.depth_um <= 0:
            raise OptimizerError(
                f"Chiplet '{self.name}' must have positive physical extents, "
                f"got ({self.width_um} x {self.depth_um} um)"
            )
        if self.power_w < 0:
            raise OptimizerError(f"Chiplet '{self.name}' power_w must be >= 0, got {self.power_w}")

    @property
    def area_um2(self) -> float:
        return self.width_um * self.depth_um

    @property
    def area_mm2(self) -> float:
        return self.area_um2 * 1e-6

    @property
    def power_density_w_cm2(self) -> float:
        """Areal power density in W/cm^2 -- the figure to sanity-check against
        real silicon (a datacenter GPU sits near 50-100 W/cm^2)."""
        area_cm2 = self.area_um2 * 1e-8
        return self.power_w / area_cm2 if area_cm2 > 0 else float("nan")

    def extent_vox(self, dx: float, dy: float) -> Tuple[int, int]:
        """Resolves this chiplet's physical footprint to a voxel extent on a
        mesh of spacing (dx, dy) in metres.

        Rounds UP, so a chiplet never occupies less die area than it
        physically does, and clamps to at least one voxel per axis so a block
        finer than a single cell still exists. Both roundings distort the
        modelled footprint; mesh_fidelity_report() quantifies that distortion
        rather than hiding it.
        """
        nx = max(1, int(math.ceil(self.width_um * 1e-6 / dx)))
        ny = max(1, int(math.ceil(self.depth_um * 1e-6 / dy)))
        return nx, ny


# Default catalog: realistic advanced-packaging block footprints, 36.0 W over
# 192 mm^2. HBM dominates on area because a real HBM3 known-good stacked die
# genuinely is ~11 x 11 mm. Per-block densities span 3.3-63 W/cm^2, which is
# the regime real parts occupy.
#
# NOTE ON DIE SIZE: this catalog needs 192 mm^2 and therefore does NOT fit on
# Stage 1's default 40x40 grid at dx=25 um, which is a 1.00 mm^2 die. That is
# the intended consequence of physical authoring -- the optimizer now raises a
# clear OptimizerError naming both areas instead of silently rescaling the
# chiplets to whatever mesh it was handed. To model a ~20 x 20 mm package,
# choose nx*dx and ny*dy accordingly, e.g.
#     --nx 40 --ny 40 --dx-um 500 --dy-um 500
DEFAULT_CHIPLET_CATALOG: Tuple[ChipletSpec, ...] = (
    ChipletSpec(name="CPU_Core",        width_um=4000.0,  depth_um=4000.0,  power_w=10.0),
    ChipletSpec(name="GPU_Core",        width_um=6000.0,  depth_um=5000.0,  power_w=14.0),
    ChipletSpec(name="HBM_Memory",      width_um=11000.0, depth_um=11000.0, power_w=4.0),
    ChipletSpec(name="NPU_Accelerator", width_um=5000.0,  depth_um=5000.0,  power_w=8.0),
)


def scale_catalog_to_total_power(
    catalog: Sequence[ChipletSpec], target_total_w: float,
) -> Tuple[ChipletSpec, ...]:
    """Rescales every block's power by one common factor so the catalog totals
    `target_total_w`, leaving footprints and the relative power split intact.

    Exists because Stage 1 asks the user for a die power and Stage 2 then used
    DEFAULT_CHIPLET_CATALOG's 36.0 W regardless -- a 3x mismatch at the 12 W
    default, i.e. two different chips in one invocation, with energy-balance
    floors 60 K apart. Silently ignoring a number the tool explicitly asked
    for is worse than never asking.

    Scaling is uniform, so a 10/14/4/8 W split becomes 10/14/4/8 times k. That
    preserves which block is hottest, and therefore the shape of the placement
    problem. It does NOT model a real chip, whose blocks would not scale
    together -- stated here rather than left implicit.
    """
    if target_total_w < 0:
        raise OptimizerError(f"target_total_w must be >= 0, got {target_total_w}")
    current = sum(c.power_w for c in catalog)
    if current <= 0:
        raise OptimizerError(
            "Cannot rescale a catalog whose total power is zero -- the ratios are undefined."
        )
    k = target_total_w / current
    return tuple(
        ChipletSpec(name=c.name, width_um=c.width_um, depth_um=c.depth_um,
                    power_w=c.power_w * k)
        for c in catalog
    )


Layout = Tuple[Tuple[int, int], ...]   # one (x0, y0) origin per catalog entry


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OptimizationResult:
    """Outcome of one optimize() run.

    best_peak_temperature_c is always a float, never None -- the best layout
    is by construction a buildable, solved, non-overlapping layout (the
    initial layout is validated as non-overlapping before the anneal starts,
    and _consider_candidate() refuses to promote any candidate without a
    solved peak temperature)."""
    best_layout: Layout
    best_cost: float
    best_peak_temperature_c: float
    best_temperature_field_k: np.ndarray
    best_cost_history: List[float]
    current_cost_history: List[float]
    initial_layout: Layout
    initial_peak_temperature_c: float
    iterations: int
    accepted_moves: int
    rejected_moves: int
    overlap_candidates: int
    seed: int
    catalog: Tuple[ChipletSpec, ...] = field(default_factory=tuple)
    # Voxel extents the physical footprints resolved to on this run's mesh,
    # parallel to `catalog`. Carried on the result so the report can show both
    # the authored physical size and the size actually modelled.
    extents_vox: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    grid_shape: Tuple[int, int, int] = (0, 0, 0)
    die_z_range: Tuple[int, int] = (0, 0)
    voxel_size_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Carried so the report can state the convective floor P/(h*A) against
    # which the reported peak must be sanity-checked.
    h_convective: float = 0.0
    t_ambient_c: float = 25.0
    elapsed_s: float = 0.0

    @property
    def improvement_c(self) -> float:
        """Peak-temperature reduction from the initial layout, in degrees C.
        Positive means the optimizer found a cooler placement."""
        return self.initial_peak_temperature_c - self.best_peak_temperature_c

    @property
    def improvement_pct(self) -> float:
        rise_initial = self.initial_peak_temperature_c
        if abs(rise_initial) < 1e-12:
            return 0.0
        return 100.0 * self.improvement_c / rise_initial

    @property
    def total_power_w(self) -> float:
        return float(sum(c.power_w for c in self.catalog))

    @property
    def catalog_area_mm2(self) -> float:
        """Total authored (physical) chiplet area."""
        return float(sum(c.area_mm2 for c in self.catalog))

    @property
    def die_area_mm2(self) -> float:
        nx, ny, _nz = self.grid_shape
        dx, dy, _dz = self.voxel_size_m
        return (nx * dx * 1e3) * (ny * dy * 1e3)

    @property
    def predicted_convective_rise_k(self) -> float:
        """Closed-form peak rise above ambient from the boundary alone:
        P * R_conv with R_conv = 1/(h*A). The internal conduction spreading
        resistance adds to this, so the true peak sits slightly above it.
        A measured peak far from this value indicates a defect, not heat."""
        nx, ny, _nz = self.grid_shape
        dx, dy, _dz = self.voxel_size_m
        area_m2 = (nx * dx) * (ny * dy)
        if area_m2 <= 0 or self.h_convective <= 0:
            return float("nan")
        return self.total_power_w / (self.h_convective * area_m2)

    @property
    def convective_floor_ratio(self) -> float:
        """Measured peak rise divided by the closed-form convective floor.
        Should be slightly above 1.0 (the excess is spreading resistance)."""
        predicted = self.predicted_convective_rise_k
        if not np.isfinite(predicted) or predicted <= 0:
            return float("nan")
        return (self.best_peak_temperature_c - self.t_ambient_c) / predicted

    @property
    def fill_factor_pct(self) -> float:
        """Fraction of the die covered by chiplets. Low fill leaves the
        annealer room to move; near-100% fill means placement is effectively
        forced and the search has little to find."""
        die = self.die_area_mm2
        return 100.0 * self.catalog_area_mm2 / die if die > 0 else 0.0

    def summarize(self) -> str:
        """Human-readable multi-line summary, including an explicit note on
        the physical scale of the reported temperatures."""
        nx, ny, nz = self.grid_shape
        dx, dy, _dz = self.voxel_size_m
        die_w_mm, die_h_mm = nx * dx * 1e3, ny * dy * 1e3
        area_cm2 = (nx * dx * 1e2) * (ny * dy * 1e2)
        density = self.total_power_w / area_cm2 if area_cm2 > 0 else float("nan")

        lines = [
            "=" * 64,
            "STAGE 2 :: CHIPLET PLACEMENT OPTIMIZATION RESULTS",
            "=" * 64,
            f"  Chiplets placed:        {len(self.catalog)} "
            f"({', '.join(c.name for c in self.catalog)})",
            f"  Total power:            {self.total_power_w:.2f} W",
            f"  Die:                    {die_w_mm:.2f} x {die_h_mm:.2f} mm "
            f"({self.die_area_mm2:.1f} mm^2) at dx={dx*1e6:.1f} dy={dy*1e6:.1f} um",
            f"  Chiplet area / fill:    {self.catalog_area_mm2:.1f} mm^2 "
            f"({self.fill_factor_pct:.1f}% of die)",
            f"  SA iterations:          {self.iterations} "
            f"({self.accepted_moves} accepted, {self.rejected_moves} rejected, "
            f"{self.overlap_candidates} overlapping)",
            f"  Initial peak:           {self.initial_peak_temperature_c:.2f} degC",
            f"  Optimized peak:         {self.best_peak_temperature_c:.2f} degC",
            # Labelled as a delta, with an explicit arrow, because this line
            # has been misread as a peak temperature more than once -- and a
            # 39 K "peak" then fails the P/(h*A) floor and looks like a solver
            # bug. The units are written as "K cooler" rather than "degC" so
            # it cannot be mistaken for an absolute temperature at a glance.
            f"  Improvement (delta):    {self.improvement_c:.2f} K cooler "
            f"({self.improvement_pct:.2f}%)  [{self.initial_peak_temperature_c:.2f} "
            f"-> {self.best_peak_temperature_c:.2f} degC]",
            f"  Optimizer wall-clock:   {self.elapsed_s:.2f} s",
            "-" * 64,
            "  SCALE CHECK (never let the headline number pass unexamined):",
            f"    areal power density   = {density:,.1f} W/cm^2",
            f"    R_conv = 1/(h*A)      = {1.0/(self.h_convective * (nx*dx) * (ny*dy)):,.3f} K/W"
            if self.h_convective > 0 and nx * dx * ny * dy > 0 else
            "    R_conv = 1/(h*A)      = n/a",
            f"    P/(h*A) predicted rise= {self.predicted_convective_rise_k:,.1f} K",
            f"    measured rise         = {self.best_peak_temperature_c - self.t_ambient_c:,.1f} K",
            f"    ratio (should be >1)  = {self.convective_floor_ratio:.4f}"
            f"   <- excess over the floor is conduction spreading resistance",
            "    NOTE: 'Improvement' above is the DIFFERENCE between two peaks,",
            "    not a rise above ambient. Never feed it into P*R_conv.",
            "=" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class ChipletPlacementOptimizer:
    """
    Simulated-annealing search over non-overlapping chiplet XY placements,
    minimizing steady-state peak temperature.

    The vertical stack (interposer / bump+underfill composite / Si die) is
    fixed and identical to the one main.py's build_chiplet_stack() assembles;
    only the XY origins of the chiplets vary. Chiplet power is injected over
    the full die z-range.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        dy: float,
        dz: float,
        catalog: Sequence[ChipletSpec] = DEFAULT_CHIPLET_CATALOG,
        t_ambient_c: float = 25.0,
        h_convective: float = 1000.0,
        bottom_bc: str = "robin",
        stack: Optional["StackSection"] = None,
        placement_pitch_um: float = DEFAULT_PLACEMENT_PITCH_UM,
        seed: int = 0,
    ):
        if not catalog:
            raise OptimizerError("Chiplet catalog is empty; nothing to place.")
        for spec in catalog:
            spec.validate()

        self.catalog: Tuple[ChipletSpec, ...] = tuple(catalog)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)
        self.t_ambient_c = float(t_ambient_c)
        self.h_convective = float(h_convective)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

        self.placement_pitch_um = float(placement_pitch_um)
        if self.placement_pitch_um <= 0:
            raise OptimizerError(
                f"placement_pitch_um must be > 0, got {self.placement_pitch_um}"
            )

        # --- Vertical stack geometry (mirrors main.build_chiplet_stack) ---
        # Layer interfaces come from PHYSICAL thicknesses in micrometers,
        # resolved against this mesh's dz. Authoring them as fractions of nz
        # made the composite layer's realized thickness a rounding artifact of
        # nz (a -16.7%..+25.0% swing), and that layer carries the stack's
        # dominant vertical resistance -- so refining the mesh silently
        # changed the geometry. See StackSection's docstring for the measured
        # table.
        self.stack = stack if stack is not None else StackSection()
        try:
            z_interposer_end, z_bump_end = self.stack.resolve_layer_boundaries(self.nz, self.dz)
        except ConfigError as exc:
            raise OptimizerError(f"Stack layer resolution failed: {exc}") from exc
        self.realized_layers_um = self.stack.realized_thicknesses_um(self.nz, self.dz)

        self.die_z_start, self.die_z_end = z_bump_end, self.nz
        if self.die_z_start >= self.die_z_end:
            raise OptimizerError(
                f"Degenerate die layer z[{self.die_z_start}:{self.die_z_end}] for nz={self.nz} "
                f"at dz={self.dz*1e6:.2f} um; increase nz or reduce the authored layer "
                f"thicknesses ({self.stack.interposer_um:.0f}/{self.stack.bump_um:.0f}/"
                f"{self.stack.die_um:.0f} um)."
            )

        # --- Resolve physical footprints to voxel extents on THIS mesh ---
        # The single point where mesh resolution enters chiplet geometry.
        self._extents: Tuple[Tuple[int, int], ...] = tuple(
            spec.extent_vox(self.dx, self.dy) for spec in self.catalog
        )

        # --- Feasibility, checked on PHYSICAL area, not voxel counts ---
        die_w_um = self.nx * self.dx * 1e6
        die_h_um = self.ny * self.dy * 1e6
        die_area_mm2 = (die_w_um * die_h_um) * 1e-6

        for spec in self.catalog:
            if spec.width_um > die_w_um or spec.depth_um > die_h_um:
                raise OptimizerError(
                    f"Chiplet '{spec.name}' ({spec.width_um:.0f} x {spec.depth_um:.0f} um) does "
                    f"not fit within the {die_w_um:.0f} x {die_h_um:.0f} um die "
                    f"({self.nx}x{self.ny} voxels at dx={self.dx*1e6:.1f} dy={self.dy*1e6:.1f} um)."
                )

        catalog_area_mm2 = sum(s.area_mm2 for s in self.catalog)
        if catalog_area_mm2 > die_area_mm2:
            raise OptimizerError(
                f"Chiplet catalog needs {catalog_area_mm2:.2f} mm^2 of die area but the die is "
                f"only {die_area_mm2:.2f} mm^2 ({die_w_um:.0f} x {die_h_um:.0f} um = "
                f"{self.nx}x{self.ny} voxels at dx={self.dx*1e6:.1f} dy={self.dy*1e6:.1f} um). "
                f"Enlarge the die via nx*dx / ny*dy, or use a smaller catalog."
            )

        # Rounding footprints up to whole voxels inflates the occupied area;
        # if the mesh is so coarse that this materially distorts the layout,
        # say so rather than letting it pass silently.
        self._warn_on_coarse_mesh()

        # --- Placement lattice (see DEFAULT_PLACEMENT_PITCH_UM) ---
        # Origins are snapped to a physical pitch so a layout found on one
        # mesh is exactly representable on any coarser mesh in a refinement
        # ladder. Resolved to voxels here, AFTER the catalog-fit checks above:
        # "these chiplets do not fit on this die" is the more fundamental
        # diagnosis and should surface first.
        self._pitch_x = max(1, int(round(self.placement_pitch_um * 1e-6 / self.dx)))
        self._pitch_y = max(1, int(round(self.placement_pitch_um * 1e-6 / self.dy)))
        if self._pitch_x > self.nx // 2 or self._pitch_y > self.ny // 2:
            raise OptimizerError(
                f"placement_pitch_um={self.placement_pitch_um:.1f} um is too coarse for this "
                f"die: it is {self._pitch_x}x{self._pitch_y} voxels on a {self.nx}x{self.ny} "
                f"grid ({self.nx*self.dx*1e6:.0f} x {self.ny*self.dy*1e6:.0f} um), leaving "
                f"almost no placeable positions. Use a pitch of at most "
                f"{min(self.nx // 2 * self.dx, self.ny // 2 * self.dy) * 1e6:.0f} um here, or "
                f"model a larger die."
            )

        realized_pitch_x_um = self._pitch_x * self.dx * 1e6
        realized_pitch_y_um = self._pitch_y * self.dy * 1e6
        if (abs(realized_pitch_x_um - self.placement_pitch_um) > 1e-9
                or abs(realized_pitch_y_um - self.placement_pitch_um) > 1e-9):
            logger.warning(
                "Placement pitch %.1f um is not a whole number of voxels at dx=%.2f "
                "dy=%.2f um; snapping to %.1f x %.1f um instead. Layouts found here may "
                "not be exactly representable on a coarser mesh.",
                self.placement_pitch_um, self.dx * 1e6, self.dy * 1e6,
                realized_pitch_x_um, realized_pitch_y_um,
            )

        # --- Build the grid + solver ONCE (see module docstring) ---
        grid_config = {
            "grid": {"nx": self.nx, "ny": self.ny, "nz": self.nz,
                     "dx": self.dx, "dy": self.dy, "dz": self.dz},
            "boundary_conditions": {
                "t_ambient_k": self.t_ambient_c + 273.15,
                "h_convective": float(h_convective),
                "bottom_bc": bottom_bc,
                "top_bc": "neumann",
                "side_bc": "neumann",
            },
        }
        try:
            gb = GridBuilder(config=grid_config)
            gb.assign_layer(slice(0, z_interposer_end), "silicon")
            gb.assign_layer(slice(z_interposer_end, z_bump_end), "bump_underfill_composite")
            gb.assign_layer(slice(z_bump_end, self.nz), "silicon")
            gb.tag_boundaries()
            gb.validate_grid()
        except GridBuilderError as exc:
            raise OptimizerError(f"Optimizer grid assembly failed: {exc}") from exc

        self.grid_builder = gb
        try:
            self.solver = ThermalSolver.from_grid_builder(gb)
        except ThermalSolverError as exc:
            raise OptimizerError(f"Optimizer solver initialization failed: {exc}") from exc

        self._cell_volume = self.dx * self.dy * self.dz
        self._die_thickness_vox = self.die_z_end - self.die_z_start

        # Elitist best-tracker state, reset at the start of every optimize().
        self._best_cost: float = math.inf
        self._best_peak_temperature_c: Optional[float] = None
        self._best_layout: Optional[Layout] = None
        self._best_field_k: Optional[np.ndarray] = None

        logger.info(
            "ChipletPlacementOptimizer ready: grid=%dx%dx%d, die z[%d:%d], "
            "%d chiplets totalling %.2f W, seed=%d",
            self.nx, self.ny, self.nz, self.die_z_start, self.die_z_end,
            len(self.catalog), sum(c.power_w for c in self.catalog), self.seed,
        )

    # ------------------------------------------------------------------
    # Mesh fidelity
    # ------------------------------------------------------------------
    def mesh_fidelity_report(self) -> List[Dict[str, object]]:
        """Per-chiplet comparison of the physical footprint against the voxel
        footprint actually modelled on this mesh.

        `area_error_pct` is how much larger the meshed block is than the real
        one, purely from rounding each axis up to a whole number of voxels.
        It shrinks as dx/dy shrink, which is the correct direction: refining
        the mesh converges the modelled geometry onto the physical geometry
        instead of changing it.
        """
        rows: List[Dict[str, object]] = []
        for spec, (nvx, nvy) in zip(self.catalog, self._extents):
            meshed_um2 = (nvx * self.dx * 1e6) * (nvy * self.dy * 1e6)
            err = 100.0 * (meshed_um2 - spec.area_um2) / spec.area_um2 if spec.area_um2 > 0 else 0.0
            rows.append({
                "name": spec.name,
                "physical_um": (spec.width_um, spec.depth_um),
                "extent_vox": (nvx, nvy),
                "meshed_um": (nvx * self.dx * 1e6, nvy * self.dy * 1e6),
                "area_error_pct": err,
                "power_density_w_cm2": spec.power_density_w_cm2,
            })
        return rows

    def _warn_on_coarse_mesh(self, threshold_pct: float = 10.0) -> None:
        """Logs a warning for any chiplet whose meshed footprint overstates
        its physical footprint by more than `threshold_pct`, and for any
        chiplet resolved to a single voxel on an axis."""
        for row, (nvx, nvy) in zip(self.mesh_fidelity_report(), self._extents):
            if nvx == 1 or nvy == 1:
                logger.warning(
                    "Chiplet '%s' resolves to %dx%d voxel(s) on this mesh -- it is at or below "
                    "the cell size, so its shape is not represented. Refine dx/dy.",
                    row["name"], nvx, nvy,
                )
            elif row["area_error_pct"] > threshold_pct:
                logger.warning(
                    "Chiplet '%s' meshed footprint overstates its physical area by %.1f%% "
                    "(%.0fx%.0f um -> %dx%d voxels = %.0fx%.0f um). Refine dx/dy for a "
                    "closer fit.",
                    row["name"], row["area_error_pct"],
                    row["physical_um"][0], row["physical_um"][1], nvx, nvy,
                    row["meshed_um"][0], row["meshed_um"][1],
                )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_pipeline_config(
        cls,
        config,
        catalog: Sequence[ChipletSpec] = DEFAULT_CHIPLET_CATALOG,
        seed: int = 0,
    ) -> "ChipletPlacementOptimizer":
        """
        Builds an optimizer from the same PipelineConfig main.py already
        assembled for Stage 1, reusing its grid, boundary-condition and stack
        parameters verbatim.

        Reads only grid / boundary_conditions / stack fields. It deliberately
        does NOT touch config.solver: the optimizer runs steady-state solves
        and has no use for dt, total_time_s, method or safety_factor, so it
        never depends on those fields existing.
        """
        return cls(
            nx=config.grid.nx, ny=config.grid.ny, nz=config.grid.nz,
            dx=config.grid.dx, dy=config.grid.dy, dz=config.grid.dz,
            catalog=catalog,
            t_ambient_c=config.boundary_conditions.t_ambient_c,
            h_convective=config.boundary_conditions.h_convective,
            bottom_bc=config.boundary_conditions.bottom_bc,
            stack=config.stack,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Layout geometry
    # ------------------------------------------------------------------
    def shelf_packed_layout(self) -> Layout:
        """
        Deterministic non-overlapping seed layout via first-fit shelf packing:
        chiplets are placed left-to-right in rows ("shelves"), each shelf as
        tall as its tallest member, wrapping to a new shelf on overflow.

        Guaranteed overlap-free by construction, which is what lets optimize()
        start from a layout that is known-buildable and known-solvable.
        """
        order = sorted(range(len(self.catalog)),
                       key=lambda i: self._extents[i][1], reverse=True)
        placements: Dict[int, Tuple[int, int]] = {}
        cursor_x, shelf_y, shelf_h = 0, 0, 0

        for idx in order:
            spec = self.catalog[idx]
            nvx, nvy = self._extents[idx]
            if cursor_x + nvx > self.nx:              # wrap to a new shelf
                shelf_y += shelf_h
                cursor_x, shelf_h = 0, 0
            # Both cursor_x and shelf_y are advanced in whole lattice steps
            # below, so every placement here is already lattice-aligned.
            if shelf_y + nvy > self.ny:
                raise OptimizerError(
                    f"Cannot shelf-pack the catalog into a {self.nx}x{self.ny} voxel die "
                    f"({self.nx*self.dx*1e6:.0f} x {self.ny*self.dy*1e6:.0f} um): "
                    f"'{spec.name}' ({spec.width_um:.0f} x {spec.depth_um:.0f} um = "
                    f"{nvx}x{nvy} voxels) overflows at shelf y={shelf_y}. The catalog fits by "
                    f"total area but not by this packing; enlarge the die or shrink the catalog."
                )
            # Advance the cursor to the next lattice position, so the seed
            # layout is itself representable on a coarser mesh. For the
            # default catalog every extent already happens to be a whole
            # number of lattice steps, but that is a property of that
            # catalog, not a guarantee -- round up explicitly.
            placements[idx] = (cursor_x, shelf_y)
            cursor_x = -(-(cursor_x + nvx) // self._pitch_x) * self._pitch_x
            shelf_h = max(shelf_h, -(-nvy // self._pitch_y) * self._pitch_y)

        return tuple(placements[i] for i in range(len(self.catalog)))

    def overlap_cell_count(self, layout: Layout) -> int:
        """Total pairwise-overlapping XY area in voxels. Zero iff the layout
        is physically buildable. Counts each overlapping pair once, so a
        triple stack contributes its three pairwise intersections."""
        total = 0
        n = len(self.catalog)
        for a in range(n):
            ax0, ay0 = layout[a]
            ax1, ay1 = ax0 + self._extents[a][0], ay0 + self._extents[a][1]
            for b in range(a + 1, n):
                bx0, by0 = layout[b]
                bx1, by1 = bx0 + self._extents[b][0], by0 + self._extents[b][1]
                ox = min(ax1, bx1) - max(ax0, bx0)
                oy = min(ay1, by1) - max(ay0, by0)
                if ox > 0 and oy > 0:
                    total += ox * oy
        return total

    def is_valid(self, layout: Layout) -> bool:
        """True iff every chiplet is in bounds and no two overlap."""
        for (nvx, nvy), (x0, y0) in zip(self._extents, layout):
            if x0 < 0 or y0 < 0 or x0 + nvx > self.nx or y0 + nvy > self.ny:
                return False
        return self.overlap_cell_count(layout) == 0

    # ------------------------------------------------------------------
    # Candidate evaluation
    # ------------------------------------------------------------------
    def _apply_layout_to_q_map(self, layout: Layout) -> None:
        """Rewrites the solver's volumetric power map in place for `layout`.

        Materials, boundary conditions and therefore the assembled sparse
        conduction operator are untouched -- only q_map changes when a
        chiplet moves, and solve_steady_state() re-reads q_map on every call.
        """
        self.solver.q_map[:] = 0.0
        zs = slice(self.die_z_start, self.die_z_end)
        for spec, (nvx, nvy), (x0, y0) in zip(self.catalog, self._extents, layout):
            # Total power is conserved exactly: the block's watts are spread
            # over the voxels it actually occupies on this mesh. Rounding the
            # footprint up therefore lowers the modelled volumetric density
            # slightly rather than inventing or losing power.
            n_cells = nvx * nvy * self._die_thickness_vox
            q_density = spec.power_w / (n_cells * self._cell_volume)
            self.solver.q_map[x0:x0 + nvx, y0:y0 + nvy, zs] += q_density

    def evaluate(self, layout: Layout) -> Tuple[float, Optional[float], Optional[np.ndarray]]:
        """
        Scores one candidate layout.

        Returns (cost, peak_temperature_c, temperature_field_k).

        For a buildable (non-overlapping, in-bounds) layout the cost IS the
        steady-state peak temperature in degrees C, and all three values are
        real.

        For an unbuildable layout there is no temperature field to compute --
        the geometry does not exist -- so the solver is never invoked and
        both peak_temperature_c and temperature_field_k are None. The cost is
        a large overlap penalty so the annealer can still rank it. Callers
        MUST NOT infer validity from cost alone; key on peak_temperature_c
        being non-None.
        """
        overlap = self.overlap_cell_count(layout)
        in_bounds = all(
            0 <= x0 and 0 <= y0
            and x0 + nvx <= self.nx and y0 + nvy <= self.ny
            for (nvx, nvy), (x0, y0) in zip(self._extents, layout)
        )
        if overlap > 0 or not in_bounds:
            penalty = OVERLAP_PENALTY_C + overlap * OVERLAP_PENALTY_PER_CELL_C
            return penalty, None, None

        self._apply_layout_to_q_map(layout)
        try:
            field_k = self.solver.solve_steady_state(apply_result=False)
        except ThermalSolverError as exc:
            if "converge" in str(exc).lower():
                raise OptimizerError(self._diagnose_convergence_failure(exc)) from exc
            raise OptimizerError(f"Steady-state evaluation failed for layout {layout}: {exc}") from exc

        peak_c = float(np.max(field_k)) - 273.15
        return peak_c, peak_c, field_k

    def _diagnose_convergence_failure(self, exc: Exception) -> str:
        """Turns a bare convergence failure into an actionable message.

        This path is now RARE. It used to fire on every package-scale grid,
        because the steady-state solve fell to a Jacobi-preconditioned
        BiCGSTAB that could not handle an anisotropic mesh. That was fixed by
        switching to conjugate gradients on the exactly-symmetric SPD form of
        the operator (see STEADY-STATE BACKEND in thermal_solver.py), and CG
        converges at every size and aspect ratio measured, up to 50:1 and
        N=307,200.

        So reaching here means something beyond the previously-known failure:
        an extreme material contrast, a pathological aspect ratio past
        anything measured, or a genuinely near-singular operator. The message
        says so rather than repeating the old advice, which would now send
        the reader after a bug that no longer exists.
        """
        n_unknowns = self.nx * self.ny * self.nz
        aspect = max(self.dx, self.dy) / self.dz if self.dz > 0 else float("nan")
        backend = ("direct sparse LU" if n_unknowns <= DIRECT_SOLVE_MAX_UNKNOWNS
                   else "conjugate gradients on the symmetrized SPD system")
        return (
            f"Steady-state solve did not converge on a {self.nx}x{self.ny}x{self.nz} grid "
            f"({n_unknowns:,} unknowns, mesh aspect ratio {aspect:.1f}:1) using the "
            f"{backend} backend. CG on the symmetrized system is known to converge at "
            f"aspect ratios up to 50:1 and up to N=307,200, so this is NOT the previously "
            f"known BiCGSTAB anisotropy failure -- that one is fixed. Suspect instead an "
            f"extreme material contrast, an aspect ratio beyond anything measured, or a "
            f"near-singular operator (verify a Robin/Dirichlet boundary is present and "
            f"h_convective is nonzero). Reducing the grid below "
            f"{DIRECT_SOLVE_MAX_UNKNOWNS:,} unknowns routes to the direct factorization, "
            f"which is a useful cross-check on whether the operator or the solver is at "
            f"fault. Underlying solver error: {exc}"
        )

    def _consider_candidate(
        self,
        cand_layout: Layout,
        cand_cost: float,
        cand_peak: Optional[float],
        cand_field: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Elitist best-tracker. Promotes `cand_layout` to the incumbent best
        only if it is BOTH cheaper than the current best AND actually
        buildable.

        The `cand_peak is not None` conjunct is the correctness invariant,
        not an optimization. evaluate() returns a finite cost for overlapping
        layouts (so the annealer can rank them) but no peak temperature (so
        there is nothing to report). Testing cost alone would let an
        unbuildable layout win whenever its penalty happened to fall below a
        genuinely hot valid incumbent, leaving best_peak_temperature_c as
        None and leaking that None into the report renderer and main.py's
        result summary.

        Returns True if the candidate became the new best.
        """
        improved = (cand_peak is not None) and (cand_cost < self._best_cost)
        if improved:
            self._best_cost = cand_cost
            self._best_peak_temperature_c = cand_peak
            self._best_layout = tuple(cand_layout)
            if cand_field is not None:
                self._best_field_k = cand_field
        return improved

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------
    def _propose(self, layout: Layout, max_jump: int) -> Layout:
        """
        Proposes a neighbour layout. With probability 0.25 (and only when
        there are at least two chiplets) it swaps two chiplets' origins,
        which lets the annealer exchange a hot chiplet with a cool one in a
        single move; otherwise it jitters one chiplet's origin by up to
        +/-max_jump voxels per axis, clamped to stay in bounds.

        Clamping to bounds means proposals are always in-bounds; they may
        still overlap, and those are penalized in evaluate().
        """
        mutable = list(layout)
        n = len(mutable)

        if n >= 2 and self._rng.random() < 0.25:
            i, j = self._rng.sample(range(n), 2)
            xi, yi = mutable[i]
            xj, yj = mutable[j]
            mutable[i] = self._snap(xj, yj, i)
            mutable[j] = self._snap(xi, yi, j)
            return tuple(mutable)

        idx = self._rng.randrange(n)
        x0, y0 = mutable[idx]
        # Jitter in whole lattice steps, so a proposal can never leave the
        # placement lattice in the first place.
        steps_x = max(1, max_jump // self._pitch_x)
        steps_y = max(1, max_jump // self._pitch_y)
        dx_move = self._rng.randint(-steps_x, steps_x) * self._pitch_x
        dy_move = self._rng.randint(-steps_y, steps_y) * self._pitch_y
        mutable[idx] = self._snap(x0 + dx_move, y0 + dy_move, idx)
        return tuple(mutable)

    def _snap(self, x: int, y: int, idx: int) -> Tuple[int, int]:
        """Snaps one origin onto the placement lattice and clamps it in bounds.

        Clamping happens AFTER snapping and snaps downward, so a clamped
        origin stays on the lattice rather than being pushed off it by the
        bound -- otherwise blocks pressed against the far edge would sit at
        off-lattice positions and break representability on a coarser mesh.
        """
        nvx, nvy = self._extents[idx]
        sx = int(round(x / self._pitch_x)) * self._pitch_x
        sy = int(round(y / self._pitch_y)) * self._pitch_y
        sx = max(0, min(sx, ((self.nx - nvx) // self._pitch_x) * self._pitch_x))
        sy = max(0, min(sy, ((self.ny - nvy) // self._pitch_y) * self._pitch_y))
        return sx, sy

    def is_on_placement_lattice(self, layout: Layout) -> bool:
        """True iff every origin lies on the placement lattice -- i.e. the
        layout is exactly representable on any mesh whose pitch divides
        placement_pitch_um."""
        return all(x % self._pitch_x == 0 and y % self._pitch_y == 0
                   for x, y in layout)

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def optimize(
        self,
        iterations: int = 30,
        initial_layout: Optional[Layout] = None,
        max_jump: Optional[int] = None,
        verbose: bool = True,
    ) -> OptimizationResult:
        """
        Runs simulated annealing for `iterations` proposals and returns the
        best buildable layout found.

        The annealing temperature schedule is geometric and scaled to the
        magnitude of the initial cost, so it adapts automatically to the
        (very large) absolute temperatures this model produces rather than
        assuming an O(1) objective.

        best_cost_history[k] is the incumbent best cost after proposal k. It
        is monotonically non-increasing by construction, because the
        incumbent only ever moves downward.
        """
        if iterations < 1:
            raise OptimizerError(f"iterations must be >= 1, got {iterations}")
        if max_jump is None:
            max_jump = max(2, min(self.nx, self.ny) // 4)

        t_start = time.perf_counter()

        # Reset elitist state so repeated optimize() calls are independent.
        self._best_cost = math.inf
        self._best_peak_temperature_c = None
        self._best_layout = None
        self._best_field_k = None

        layout = tuple(initial_layout) if initial_layout is not None else self.shelf_packed_layout()
        if not self.is_valid(layout):
            raise OptimizerError(
                f"Initial layout {layout} is invalid (out of bounds or overlapping); "
                f"optimize() requires a buildable starting point."
            )

        cur_cost, cur_peak, cur_field = self.evaluate(layout)
        if cur_peak is None:
            raise OptimizerError(
                "Initial layout evaluated to no peak temperature despite passing "
                "is_valid(); this indicates an evaluate()/is_valid() disagreement."
            )
        initial_layout_used, initial_peak = layout, cur_peak
        self._consider_candidate(layout, cur_cost, cur_peak, cur_field)

        # Geometric schedule from 5% of the initial cost down to 0.1% of it.
        temp0 = max(1.0, 0.05 * abs(cur_cost))
        temp_end = max(1e-3, 1e-3 * abs(cur_cost))
        cooling = (temp_end / temp0) ** (1.0 / max(1, iterations - 1)) if iterations > 1 else 1.0

        best_cost_history: List[float] = []
        current_cost_history: List[float] = []
        accepted = rejected = overlap_candidates = 0

        if verbose:
            print(f"[optimizer] Annealing {iterations} proposals from initial peak "
                  f"{initial_peak:.2f} degC (seed={self.seed}, max_jump={max_jump})")

        temp = temp0
        for k in range(iterations):
            cand_layout = self._propose(layout, max_jump)
            cand_cost, cand_peak, cand_field = self.evaluate(cand_layout)
            if cand_peak is None:
                overlap_candidates += 1

            delta = cand_cost - cur_cost
            if delta < 0:
                accept = True
            elif temp <= 0:
                accept = False
            else:
                accept = self._rng.random() < math.exp(-delta / temp)

            if accept:
                layout, cur_cost, cur_peak, cur_field = cand_layout, cand_cost, cand_peak, cand_field
                accepted += 1
            else:
                rejected += 1

            self._consider_candidate(cand_layout, cand_cost, cand_peak, cand_field)

            best_cost_history.append(self._best_cost)
            current_cost_history.append(cur_cost)
            temp *= cooling

            if verbose and (k + 1) % 10 == 0:
                print(f"[optimizer]   iter {k + 1:>4d}/{iterations}  "
                      f"best={self._best_cost:>12.2f} degC  "
                      f"current={cur_cost:>12.2f}  temp={temp:>10.2f}")

        # Post-conditions: the tracker must have produced a real, buildable best.
        if self._best_layout is None or self._best_peak_temperature_c is None:
            raise OptimizerError(
                "Optimization finished without a valid best layout. This should be "
                "unreachable -- the initial layout is validated and seeded into the "
                "tracker before annealing begins."
            )
        if not self.is_valid(self._best_layout):
            raise OptimizerError(
                f"Best layout {self._best_layout} is not buildable; the elitist "
                f"tracker admitted an overlapping candidate."
            )

        best_field = self._best_field_k
        if best_field is None:
            # Defensive: recompute rather than hand back a None field.
            _c, _p, best_field = self.evaluate(self._best_layout)

        elapsed = time.perf_counter() - t_start
        result = OptimizationResult(
            best_layout=self._best_layout,
            best_cost=float(self._best_cost),
            best_peak_temperature_c=float(self._best_peak_temperature_c),
            best_temperature_field_k=best_field,
            best_cost_history=best_cost_history,
            current_cost_history=current_cost_history,
            initial_layout=initial_layout_used,
            initial_peak_temperature_c=float(initial_peak),
            iterations=iterations,
            accepted_moves=accepted,
            rejected_moves=rejected,
            overlap_candidates=overlap_candidates,
            seed=self.seed,
            catalog=self.catalog,
            extents_vox=self._extents,
            grid_shape=(self.nx, self.ny, self.nz),
            die_z_range=(self.die_z_start, self.die_z_end),
            voxel_size_m=(self.dx, self.dy, self.dz),
            h_convective=self.h_convective,
            t_ambient_c=self.t_ambient_c,
            elapsed_s=elapsed,
        )
        logger.info(
            "optimize() done in %.2fs: %d iters, best peak %.2f degC "
            "(initial %.2f degC, improvement %.2f degC)",
            elapsed, iterations, result.best_peak_temperature_c,
            result.initial_peak_temperature_c, result.improvement_c,
        )
        return result


# ---------------------------------------------------------------------------
# HTML report
#
# House style, matching src/visualization.py: a module-level template with
# __PLACEHOLDER__ tokens substituted via str.replace() (NOT str.format(), which
# would collide with the literal braces throughout the CSS).
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; background: #0a0a0f; color: #e8e8f0;
               font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
  h1 { font-size: 24px; color: #ff9d5c; margin: 0 0 4px; font-weight: 600; }
  .sub { color: #9ea3b0; font-size: 13px; margin-bottom: 28px; }
  h2 { font-size: 16px; color: #ff9d5c; margin: 36px 0 12px; font-weight: 600;
       border-bottom: 1px solid #3a3a55; padding-bottom: 6px; }
  .cards { display: flex; flex-wrap: wrap; gap: 14px; margin: 20px 0; }
  .card { background: rgba(20,20,32,0.9); border: 1px solid #3a3a55; border-radius: 8px;
          padding: 14px 18px; min-width: 176px; flex: 1 1 176px; }
  .card .label { color: #9ea3b0; font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.06em; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .good { color: #6ee7a8; } .warn { color: #ff9d5c; } .plain { color: #e8e8f0; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 8px; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid #2a2a40; }
  th { color: #9ea3b0; font-weight: 600; font-size: 11px; text-transform: uppercase;
       letter-spacing: 0.06em; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .figure { margin: 18px 0; background: rgba(20,20,32,0.9); border: 1px solid #3a3a55;
            border-radius: 8px; padding: 14px; overflow-x: auto; }
  .figure img { display: block; width: 100%; height: auto; border-radius: 4px; }
  .note { background: rgba(255,157,92,0.08); border-left: 3px solid #ff9d5c;
          padding: 12px 16px; border-radius: 0 6px 6px 0; font-size: 13px;
          color: #d8d8e4; margin: 18px 0; }
  .note b { color: #ff9d5c; }
  footer { margin-top: 40px; padding-top: 14px; border-top: 1px solid #2a2a40;
           color: #6a6a80; font-size: 12px; }
  code { background: #1a1a28; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>__TITLE__</h1>
  <div class="sub">Stage 2 simulated-annealing chiplet placement &middot; generated __GENERATED__</div>

  <div class="cards">
    <div class="card"><div class="label">Optimized peak</div>
      <div class="value good">__BEST_PEAK__ &deg;C</div></div>
    <div class="card"><div class="label">Initial peak</div>
      <div class="value plain">__INITIAL_PEAK__ &deg;C</div></div>
    <div class="card"><div class="label">Improvement</div>
      <div class="value good">__IMPROVEMENT__ &deg;C</div></div>
    <div class="card"><div class="label">SA iterations</div>
      <div class="value plain">__ITERATIONS__</div></div>
    <div class="card"><div class="label">Total power</div>
      <div class="value plain">__TOTAL_POWER__ W</div></div>
    <div class="card"><div class="label">Optimizer time</div>
      <div class="value plain">__ELAPSED__ s</div></div>
  </div>

  <div class="note">
    <b>Scale note.</b> __SCALE_NOTE__
  </div>

  <h2>Convergence</h2>
  <div class="figure"><img alt="Simulated annealing convergence history"
       src="data:image/png;base64,__CONVERGENCE_PNG__"></div>

  <h2>Optimized placement and resulting thermal field</h2>
  <div class="figure"><img alt="Optimized chiplet placement map and die-top temperature field"
       src="data:image/png;base64,__PLACEMENT_PNG__"></div>

  <h2>Placement table</h2>
  <table>
    <thead><tr><th>Chiplet</th><th class="num">Footprint (&micro;m)</th>
      <th class="num">Meshed (vox)</th><th class="num">Power (W)</th>
      <th class="num">Density (W/cm&sup2;)</th>
      <th class="num">Optimized (x, y) mm</th><th class="num">Moved (mm)</th></tr></thead>
    <tbody>__PLACEMENT_ROWS__</tbody>
  </table>

  <h2>Run configuration</h2>
  <table>
    <tbody>__CONFIG_ROWS__</tbody>
  </table>

  <footer>
    RISC3D-AI Stage 2 &middot; objective: steady-state peak junction temperature
    via <code>ThermalSolver.solve_steady_state()</code> &middot;
    overlapping layouts are never solved and never promoted to best.
  </footer>
</div>
</body>
</html>
"""


def _fig_to_base64_png(fig, dpi: int = 110) -> str:
    """Renders a Matplotlib figure to a base64-encoded PNG string and closes
    it, so the returned HTML is fully self-contained (no sidecar image files)."""
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
    finally:
        plt.close(fig)
        buf.close()


def _style_axes(ax) -> None:
    """Applies the report's dark palette to one axes object."""
    ax.set_facecolor("#12121c")
    for spine in ax.spines.values():
        spine.set_color("#3a3a55")
    ax.tick_params(colors="#9ea3b0", labelsize=8)
    ax.xaxis.label.set_color("#e8e8f0")
    ax.yaxis.label.set_color("#e8e8f0")
    ax.title.set_color("#ff9d5c")


def _render_convergence_figure(result: OptimizationResult) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 3.6), facecolor="#0a0a0f")
    iters = np.arange(1, len(result.best_cost_history) + 1)

    current = np.array(result.current_cost_history, dtype=float)
    finite_current = current[current < OVERLAP_PENALTY_C]
    ax.plot(iters, current, color="#5c8dff", lw=1.0, alpha=0.55,
            label="Accepted current cost", zorder=2)
    ax.plot(iters, result.best_cost_history, color="#ff9d5c", lw=2.0,
            label="Incumbent best cost", zorder=3)
    ax.axhline(result.initial_peak_temperature_c, color="#6a6a80", ls="--", lw=1.0,
               label="Initial layout", zorder=1)

    ax.set_xlabel("Simulated-annealing proposal")
    ax.set_ylabel("Peak temperature (deg C)")
    ax.set_title("Convergence of steady-state peak temperature")
    # Penalty-valued (overlapping) proposals are ~1e6 and would flatten the
    # axis; clip the view to the physically meaningful band.
    if finite_current.size:
        lo = min(float(np.min(finite_current)), min(result.best_cost_history))
        hi = max(float(np.max(finite_current)), result.initial_peak_temperature_c)
        pad = max(1.0, 0.05 * (hi - lo))
        ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, color="#2a2a40", lw=0.6)
    leg = ax.legend(fontsize=8, facecolor="#12121c", edgecolor="#3a3a55", labelcolor="#e8e8f0")
    leg.get_frame().set_alpha(0.9)
    _style_axes(ax)
    return _fig_to_base64_png(fig)


def _render_placement_figure(result: OptimizationResult) -> str:
    """Two-panel figure: the optimized placement map, and the die-top
    temperature field. Both panels carry explicit x- and y-axis titles."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6), facecolor="#0a0a0f")
    nx, ny, _nz = result.grid_shape
    dx, dy, _dz = result.voxel_size_m
    # Axes are drawn in physical millimetres, not voxel indices -- the layout
    # is a physical arrangement and should read as one regardless of mesh.
    die_w_mm, die_h_mm = nx * dx * 1e3, ny * dy * 1e3

    # --- Panel 1: placement map ---
    palette = ["#ff9d5c", "#5c8dff", "#6ee7a8", "#e07be0", "#f2d16b", "#7bdcd0"]
    for i, (spec, (nvx, nvy), (x0, y0)) in enumerate(
            zip(result.catalog, result.extents_vox, result.best_layout)):
        color = palette[i % len(palette)]
        x_mm, y_mm = x0 * dx * 1e3, y0 * dy * 1e3
        w_mm, h_mm = nvx * dx * 1e3, nvy * dy * 1e3
        ax1.add_patch(Rectangle((x_mm, y_mm), w_mm, h_mm,
                                facecolor=color, alpha=0.42, edgecolor=color, lw=1.8))
        ax1.text(x_mm + w_mm / 2.0, y_mm + h_mm / 2.0,
                 f"{spec.name}\n{spec.power_w:.0f} W\n{spec.power_density_w_cm2:.0f} W/cm²",
                 ha="center", va="center", fontsize=6.5, color="#0a0a0f", weight="bold")
    ax1.set_xlim(0, die_w_mm)
    ax1.set_ylim(0, die_h_mm)
    ax1.set_aspect("equal")
    ax1.set_xlabel("Die X (mm)")
    ax1.set_ylabel("Die Y (mm)")
    ax1.set_title(f"Optimized placement ({result.fill_factor_pct:.0f}% fill)")
    ax1.grid(True, color="#2a2a40", lw=0.5)
    _style_axes(ax1)

    # --- Panel 2: die-top temperature field ---
    field_c = result.best_temperature_field_k - 273.15
    top_layer = field_c[:, :, result.grid_shape[2] - 1]
    im = ax2.imshow(top_layer.T, origin="lower", cmap="inferno",
                    extent=(0, die_w_mm, 0, die_h_mm), aspect="equal",
                    interpolation="nearest")
    ax2.set_xlabel("Die X (mm)")
    ax2.set_ylabel("Die Y (mm)")
    ax2.set_title("Die-top steady-state temperature")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("deg C", color="#e8e8f0", fontsize=8)
    cbar.ax.tick_params(colors="#9ea3b0", labelsize=7)
    cbar.outline.set_edgecolor("#3a3a55")
    _style_axes(ax2)

    fig.tight_layout()
    return _fig_to_base64_png(fig)


def render_optimization_report(
    result: OptimizationResult,
    output_dir: str = "outputs",
    filename: str = "optimized_thermal_report.html",
    title: str = "RISC3D-AI :: Optimized Chiplet Placement",
) -> str:
    """
    Writes a single self-contained HTML report for one optimize() run and
    returns its path.

    Self-contained means exactly that: both figures are embedded as base64
    PNG data URIs, so the .html file is the whole deliverable with no sidecar
    images and no network access required to view it.

    Does not open a browser -- callers that want auto-launch should call
    webbrowser.open() on the returned path, as main.py does.
    """
    if result.best_peak_temperature_c is None:
        raise OptimizerError(
            "OptimizationResult.best_peak_temperature_c is None; refusing to render a "
            "report for a layout that was never successfully solved."
        )

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as exc:
        raise OptimizerError(f"Could not create output_dir '{output_dir}': {exc}") from exc

    try:
        convergence_png = _render_convergence_figure(result)
        placement_png = _render_placement_figure(result)
    except Exception as exc:  # noqa: BLE001
        raise OptimizerError(f"Failed to render optimization report figures: {exc}") from exc

    nx, ny, nz = result.grid_shape
    dx, dy, dz = result.voxel_size_m

    rows: List[str] = []
    for spec, (nvx, nvy), (ix, iy), (bx, by) in zip(
            result.catalog, result.extents_vox, result.initial_layout, result.best_layout):
        # Distance moved reported physically, in mm, not in voxels.
        moved_mm = math.hypot((bx - ix) * dx, (by - iy) * dy) * 1e3
        rows.append(
            f"<tr><td>{spec.name}</td>"
            f"<td class='num'>{spec.width_um:,.0f} &times; {spec.depth_um:,.0f}</td>"
            f"<td class='num'>{nvx} &times; {nvy}</td>"
            f"<td class='num'>{spec.power_w:.1f}</td>"
            f"<td class='num'>{spec.power_density_w_cm2:,.1f}</td>"
            f"<td class='num'>({bx * dx * 1e3:.2f}, {by * dy * 1e3:.2f})</td>"
            f"<td class='num'>{moved_mm:.2f}</td></tr>"
        )
    config_rows = "".join(
        f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in [
            ("Grid", f"{nx} &times; {ny} &times; {nz} voxels"),
            ("Voxel size", f"{dx*1e6:.1f} &times; {dy*1e6:.1f} &times; {dz*1e6:.1f} &micro;m"),
            ("Die size", f"{nx*dx*1e3:.2f} &times; {ny*dy*1e3:.2f} mm "
                         f"({result.die_area_mm2:.1f} mm&sup2;)"),
            ("Chiplet area", f"{result.catalog_area_mm2:.1f} mm&sup2; "
                             f"({result.fill_factor_pct:.1f}% fill)"),
            ("Die layer", f"z[{result.die_z_range[0]}:{result.die_z_range[1]}]"),
            ("Objective", "steady-state peak temperature (solve_steady_state)"),
            ("SA proposals", f"{result.iterations} "
                             f"({result.accepted_moves} accepted / {result.rejected_moves} rejected)"),
            ("Overlapping proposals", f"{result.overlap_candidates} (never solved, never promoted)"),
            ("Random seed", str(result.seed)),
        ]
    )

    area_cm2 = (nx * dx * 1e2) * (ny * dy * 1e2)
    density = result.total_power_w / area_cm2 if area_cm2 > 0 else float("nan")
    # Closed-form convective floor: R_conv ~ 1/(h*A), so the peak rise above
    # ambient should sit near P/(h*A). Stated explicitly so the reported peak
    # is always checked against physics rather than merely displayed.
    predicted_rise = result.predicted_convective_rise_k
    scale_note = (
        f"The die is {nx*dx*1e3:.2f} &times; {ny*dy*1e3:.2f} mm "
        f"({result.die_area_mm2:.1f} mm&sup2;) carrying {result.total_power_w:.1f} W across "
        f"{result.catalog_area_mm2:.1f} mm&sup2; of chiplets &mdash; {result.fill_factor_pct:.0f}% "
        f"fill at an average {density:,.1f} W/cm&sup2;. "
        f"Sanity-check the headline number against the convective floor: with "
        f"R<sub>conv</sub> &asymp; 1/(h&middot;A), the peak rise above ambient should land near "
        f"P/(h&middot;A) &asymp; {predicted_rise:,.1f} K at h = {result.h_convective:,.0f} "
        f"W/m&sup2;K &mdash; measured ratio {result.convective_floor_ratio:.4f}. "
        f"A peak far from that is a defect, not merely a hot chip. Note also that "
        f"<em>improvement</em> is the difference between two peaks, not a rise above ambient, "
        f"so it must never be fed into P&middot;R<sub>conv</sub>."
    )

    html = _REPORT_TEMPLATE
    html = html.replace("__TITLE__", title)
    html = html.replace("__GENERATED__", time.strftime("%Y-%m-%d %H:%M:%S"))
    html = html.replace("__BEST_PEAK__", f"{result.best_peak_temperature_c:,.1f}")
    html = html.replace("__INITIAL_PEAK__", f"{result.initial_peak_temperature_c:,.1f}")
    html = html.replace("__IMPROVEMENT__", f"{result.improvement_c:,.1f}")
    html = html.replace("__ITERATIONS__", str(result.iterations))
    html = html.replace("__TOTAL_POWER__", f"{result.total_power_w:.1f}")
    html = html.replace("__ELAPSED__", f"{result.elapsed_s:.2f}")
    html = html.replace("__SCALE_NOTE__", scale_note)
    html = html.replace("__CONVERGENCE_PNG__", convergence_png)
    html = html.replace("__PLACEMENT_PNG__", placement_png)
    html = html.replace("__PLACEMENT_ROWS__", "".join(rows))
    html = html.replace("__CONFIG_ROWS__", config_rows)

    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
    except OSError as exc:
        raise OptimizerError(f"Failed to write optimization report to '{filepath}': {exc}") from exc

    logger.info("Wrote optimization report to %s (%d bytes)", filepath, os.path.getsize(filepath))
    return filepath


if __name__ == "__main__":
    # Smoke test at the reference configuration documented in the module
    # docstring. dx=dy=500 um over 40x40 voxels gives a 20 x 20 mm package,
    # which is what the default catalog's 192 mm^2 of real chiplets needs;
    # at dx=25 um (a 1 mm^2 die) this same catalog correctly raises
    # OptimizerError rather than silently shrinking the chiplets.
    #
    # nz=24 at dz=10 um gives the exact 100/40/100 um stack and 38,400
    # unknowns, which routes to conjugate gradients on the symmetrized SPD
    # system. That path converges here; an older comment in this spot warned
    # against nz=24 because the previous BiCGSTAB backend did not, which is
    # no longer true (see SOLVER BACKEND in the module docstring).
    opt = ChipletPlacementOptimizer(nx=40, ny=40, nz=24,
                                    dx=500e-6, dy=500e-6, dz=10e-6, seed=7)
    res = opt.optimize(iterations=30)
    print(res.summarize())
    print("Report:", render_optimization_report(res))
    print("Report:", render_optimization_report(res))