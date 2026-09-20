"""
studio/engine.py
RISC3D Studio :: the physics engine behind the app.

Every temperature the app shows comes from a full 3-D steady-state solve of
the package -- finite-volume conduction through the silicon interposer, the
microbump/underfill layer and the die, cooled through the die backside -- on
the verified solver in src/thermal_solver.py.

THE OPTIMIZER IS EXACT, NOT A HEURISTIC
It finds the placement with the lowest peak temperature among EVERY layout
on the placement grid, and proves it. Two facts make that affordable:

  1. Superposition. The steady-state equation is linear, so the temperature
     rise of any layout is exactly the sum of the rises each chiplet causes
     on its own. One solve per (chiplet, grid position) -- a single batched
     triangular solve against the cached factorization -- gives every layout.

  2. Heat only adds. Every chiplet's rise is non-negative in every cell
     (checked numerically on every run). So for a partial layout,
         max over cells of [ placed rises + least possible rise of the rest ]
     is a PROVEN lower bound on the peak of every layout that completes it.
     Whole branches whose bound is not below the best complete layout found
     so far cannot contain a better one, and are skipped without loss.

Nothing is sampled and nothing is estimated. The winner is re-solved from
scratch with the full solver, and the result is refused if the two differ by
more than RESOLVE_TOL_K.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from src.config_loader import (
    BoundaryConditionsSection, CaseSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection,
)
from src.experiment import (
    ChipletDef, Placement, PowerBlock, PowerGrid, block_weights,
    build_power_map, build_stack,
    grid_weights, package_from_config, power_z_range, solve_placement, uniform_share,
    validate_placement,
)
from src.surrogate import spreading_length_m
from src.thermal_solver import DIRECT_SOLVE_MAX_UNKNOWNS, ThermalSolver

for _logger in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder",
                "RISC3D-AI.ConfigLoader", "RISC3D-AI.Optimizer", "RISC3D-AI.Exhaustive"):
    logging.getLogger(_logger).setLevel(logging.WARNING)

# Mesh sensitivity, measured by tools/mesh_convergence.py and read from its
# output. Absent that file the app simply reports no band rather than an
# invented one.
VALIDATION_DIR = Path(__file__).resolve().parent.parent / "validation"
MESH_STUDY_PATH = VALIDATION_DIR / "mesh_convergence.json"
MATERIAL_STUDY_PATH = VALIDATION_DIR / "material_sensitivity.json"
BOUNDARY_STUDY_PATH = VALIDATION_DIR / "boundary_sensitivity.json"

DZ_M = 20e-6
DZ_UM = 20.0
STACK_UM = (100.0, 40.0, 100.0)          # the default: interposer / bumps / die
LAYER_NAMES = ("Interposer (Si)", "Microbumps + underfill", "Die (Si)")
# Each layer must hold a whole number of 20 um cells, at least one, and the
# whole stack must stay small enough to mesh laterally as well.
MIN_LAYER_UM = DZ_UM
MAX_STACK_UM = 600.0

# Assembly needs room between chiplets for underfill flow and microbump
# dispense tolerance; 100 um is a common floor for 2.5D packages. Enforced on
# the mesh, so the clearance actually applied is a whole number of cells.
DEFAULT_KEEPOUT_UM = 100.0
MAX_KEEPOUT_UM = 2000.0

# Where the power sits INSIDE a chiplet. A chiplet is not a uniform heater:
# the compute array draws most of the current over part of the die, and that
# is where the junction runs hottest. Each entry is
# (x, y, width, height, share of the chiplet's power), in fractions of the
# footprint, so a pattern survives a resize and transposes under rotation.
POWER_PATTERNS: Dict[str, Tuple[Tuple[float, float, float, float, float], ...]] = {
    "uniform": (),
    # A centred compute array: 40 % of the area carrying 70 % of the power,
    # the shape of an NPU MAC matrix or a GPU shader cluster.
    "mac_array": ((0.185, 0.185, 0.63, 0.63, 0.70),),
    # Two engines side by side, each 20 % of the area with 35 % of the power.
    "two_engines": ((0.08, 0.20, 0.34, 0.60, 0.35), (0.58, 0.20, 0.34, 0.60, 0.35)),
    # Four cores in a 2 x 2 complex, 17 % of the power each.
    "quad_cores": ((0.08, 0.08, 0.37, 0.37, 0.17), (0.55, 0.08, 0.37, 0.37, 0.17),
                   (0.08, 0.55, 0.37, 0.37, 0.17), (0.55, 0.55, 0.37, 0.37, 0.17)),
}
MAX_BLOCKS = 8
# An imported map is a grid of relative weights, one per tile. The cap is
# generous -- a 64 x 64 map is 4,096 tiles, far finer than any mesh the app
# builds -- but not unbounded, since the file comes from outside.
MAX_GRID_TILES = 8192
MAX_GRID_SIDE = 256

# THE COOLING PATH ABOVE THE DIE.
# Real hardware does not put a film coefficient on bare silicon: heat crosses
# a TIM into a lid or a cold plate, spreads sideways in that metal, and only
# then meets the coolant. The spreading is the point -- a copper lid moves
# heat away from the hotspot far better than it conducts through -- so these
# layers are meshed, not folded into an effective h.
#
# They are meshed COMPACTLY: a 1 mm lid is represented by a few 20 um cells
# with anisotropic conductivities chosen so that the layer keeps both the
# quantity that governs spreading (in-plane sheet conductance k*t) and the
# one that governs the temperature drop through it (t/k):
#     k_inplane = k * t_real / t_model      k_through = k * t_model / t_real
# That is exact for a layer whose in-plane and through-plane conduction are
# independent, which is what a thin uniform sheet is. What it gives up is the
# temperature profile INSIDE the layer, which nothing downstream needs.
# (rho is scaled the same way so the thermal mass is right if a transient
# mode ever uses these layers.)
COOLING_LAYER_FIELDS = ("name", "thickness_um", "k_w_mk", "cells")
COOLING_STACKS: Dict[str, Tuple[Tuple[str, float, float, int], ...]] = {
    # h applies straight to the die backside: a bare die on a cold plate.
    "direct": (),
    # Thermal grease to a copper lid, the common 2.5D package.
    "lid": (("TIM", 50.0, 5.0, 1), ("Copper lid", 1000.0, 400.0, 2)),
    # Lid, a second TIM and an aluminium heatsink base carrying the fins.
    "lid_heatsink": (("TIM 1", 50.0, 5.0, 1), ("Copper lid", 1000.0, 400.0, 2),
                     ("TIM 2", 100.0, 3.0, 1), ("Heatsink base", 2000.0, 200.0, 2)),
    # Cold plate bonded straight to the die with a thin high-conductivity TIM.
    "coldplate": (("TIM", 25.0, 8.0, 1), ("Cold plate base", 1500.0, 390.0, 2)),
}
MAX_COOLING_CELLS = 10
# Unknowns the app is willing to factorize, per mesh level. Above the
# protected solver's own threshold because the studio factorizes the symmetric
# form with a fill-reducing ordering.
#
# The finer level is not a different method and not an approximation: the
# factorization is exact at either size and the residual is the same 1e-12.
# What changes is cost and discretization error, both MEASURED on a 20 mm
# package (tools/mesh_convergence.py, validation/mesh_convergence.json):
#
#   0.5  mm    19,200 unknowns    0.4 s    123 MB
#   0.25 mm    76,800 unknowns    4.1 s    346 MB
#   0.25 mm    96,000 unknowns   22.9 s    493 MB   (with a cold plate meshed)
#   0.25 mm   115,200 unknowns   25.0 s    783 MB   (with a lid and heatsink)
#
# What the error is depends on the DESIGN, not just the cell size, so it is
# not a column here: see mesh_band() for the band a design is quoted and
# mesh_check() for measuring the design in hand instead of estimating it.
#
# Standard is the default because a trade-show laptop should answer in under a
# second. Fine exists because "we cannot go finer" would be false: the answer
# is exact at either size -- residuals stay at 1e-11 -- and somebody checking
# a marginal design should be able to spend the seconds and get the smaller
# discretization error instead of being told no. The fine cap is set so that
# every stack the app can build reaches 0.25 mm on a 20 mm package.
MESH_LEVELS = {"standard": 36_000, "fine": 120_000}
DEFAULT_MESH = "standard"
STUDIO_MAX_UNKNOWNS = MESH_LEVELS[DEFAULT_MESH]
# Cells the model cache will hold across all models before evicting the
# oldest. One fine model is 76,800 cells and about 640 MB of factorization,
# so a fixed count of six would reach four gigabytes.
MAX_CACHED_CELLS = 200_000
CELL_CHOICES_MM = (0.25, 0.5, 1.0, 2.0)
MAX_CHIPLETS = 16
# Root branches are searched on several cores. NumPy releases the GIL for the
# array work each node does, so threads overlap; the cap keeps a laptop
# responsive while its browser is drawing the progress overlay.
MAX_SEARCH_THREADS = 8
# Columns per thread when solving a batch of single-chiplet fields. Smaller
# chunks balance better; too small and the per-call overhead dominates.
SOLVE_CHUNK = 24
# Every threaded batch is checked by substituting it back into the system it
# claims to solve: max |S theta - q| over ALL columns. SciPy's SuperLU object
# is not documented as thread-safe, and this product does not report numbers
# it has not checked. One sparse product over the batch costs a few per cent.
THREADED_SOLVE_TOL = 1e-9

# The re-solved winner must match the superposed value to this many kelvin.
RESOLVE_TOL_K = 1e-6
# A unit rise below -tol * (its own maximum) means the bound would be invalid.
NEGATIVE_FIELD_TOL = 1e-9


class StudioError(Exception):
    """A request the engine cannot honour, with a message fit for the UI."""


class SearchCancelled(Exception):
    """The user stopped an optimization."""


def _whole(value: float, unit: float) -> Optional[int]:
    """value / unit as an int if it is one (to 1e-9 relative), else None."""
    n = value / unit
    k = round(n)
    return int(k) if abs(n - k) <= 1e-9 * max(1.0, abs(n)) else None


def _number(raw: dict, key: str, lo: float, hi: float, where: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StudioError(f"{where}: {key} must be a number")
    if not lo <= value <= hi:
        raise StudioError(f"{where}: {key} must be between {lo:g} and {hi:g}, got {value:g}")
    return float(value)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PackageSpec:
    size_mm: float          # square package side
    h_w_m2k: float          # effective heat-transfer coefficient on the die backside
    ambient_c: float
    pitch_mm: float         # placement grid for the optimizer
    stack_um: Tuple[float, float, float] = STACK_UM   # interposer / bumps / die
    keepout_um: float = DEFAULT_KEEPOUT_UM            # minimum chiplet-to-chiplet clearance
    cooling: str = "direct"                           # which path above the die
    cooling_layers: Tuple[Tuple[str, float, float, int], ...] = ()
    mesh: str = DEFAULT_MESH                          # how fine the app may go

    @staticmethod
    def parse(raw) -> "PackageSpec":
        if not isinstance(raw, dict):
            raise StudioError("package must be an object")
        stack = raw.get("stack_um")
        if stack is None:
            layers = STACK_UM
        else:
            if not isinstance(stack, (list, tuple)) or len(stack) != 3:
                raise StudioError("stack_um must be three thicknesses: interposer, bumps, die")
            names = ("interposer", "bumps", "die")
            layers = tuple(_number({"t": value}, "t", MIN_LAYER_UM, MAX_STACK_UM, f"stack ({name})")
                           for name, value in zip(names, stack))
            for name, value in zip(names, layers):
                if _whole(value, DZ_UM) is None:
                    raise StudioError(f"the {name} layer ({value:g} um) must be a whole number "
                                      f"of {DZ_UM:g} um cells")
            if sum(layers) > MAX_STACK_UM:
                raise StudioError(f"the stack totals {sum(layers):g} um; "
                                  f"at most {MAX_STACK_UM:g} um is supported")
        cooling, cooling_layers = PackageSpec._parse_cooling(raw)
        mesh = raw.get("mesh", DEFAULT_MESH)
        if not isinstance(mesh, str) or mesh not in MESH_LEVELS:
            raise StudioError(f"mesh must be one of {', '.join(sorted(MESH_LEVELS))}")
        spec = PackageSpec(
            size_mm=_number(raw, "size_mm", 5.0, 40.0, "package"),
            h_w_m2k=_number(raw, "h_w_m2k", 10.0, 1e6, "package"),
            ambient_c=_number(raw, "ambient_c", -40.0, 125.0, "package"),
            pitch_mm=_number(raw, "pitch_mm", 0.5, 20.0, "package"),
            stack_um=layers,
            keepout_um=(DEFAULT_KEEPOUT_UM if raw.get("keepout_um") is None
                        else _number(raw, "keepout_um", 0.0, MAX_KEEPOUT_UM, "package")),
            cooling=cooling,
            cooling_layers=cooling_layers,
            mesh=mesh,
        )
        cell = spec.cell_mm
        if _whole(spec.pitch_mm, cell) is None:
            raise StudioError(f"the placement grid ({spec.pitch_mm:g} mm) must be a whole "
                              f"number of {cell:g} mm mesh cells")
        return spec

    @staticmethod
    def _parse_cooling(raw: dict) -> Tuple[str, Tuple[Tuple[str, float, float, int], ...]]:
        """The path between the die backside and the coolant.

        Either a named stack, or explicit layers so a design file can carry a
        path the presets do not cover.
        """
        name = raw.get("cooling", "direct")
        if not isinstance(name, str) or (name not in COOLING_STACKS and name != "custom"):
            raise StudioError(f"cooling must be one of {', '.join(sorted(COOLING_STACKS))}")
        explicit = raw.get("cooling_layers")
        if explicit is None:
            return name, COOLING_STACKS.get(name, ())
        if not isinstance(explicit, list):
            raise StudioError("cooling_layers must be a list")
        layers = []
        for n, item in enumerate(explicit, start=1):
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                raise StudioError(f"cooling layer {n} must be "
                                  f"[name, thickness_um, k_w_mk, cells]")
            label, thickness, k, cells = item
            label = str(label)[:40]
            thickness = _number({"t": thickness}, "t", 1.0, 20000.0, f"cooling layer {n}")
            k = _number({"k": k}, "k", 0.05, 2000.0, f"cooling layer {n}")
            if not isinstance(cells, int) or isinstance(cells, bool) or not 1 <= cells <= 6:
                raise StudioError(f"cooling layer {n}: cells must be a whole number, 1 to 6")
            layers.append((label, thickness, k, cells))
        if sum(layer[3] for layer in layers) > MAX_COOLING_CELLS:
            raise StudioError(f"the cooling path may use at most {MAX_COOLING_CELLS} mesh cells")
        return ("custom" if name == "direct" and layers else name), tuple(layers)

    @property
    def cooling_cells(self) -> int:
        return sum(layer[3] for layer in self.cooling_layers)

    @property
    def die_cells(self) -> int:
        """Cells of real die, before anything the cooling path adds."""
        return int(round(self.stack_um[2] / DZ_UM))

    @property
    def nz(self) -> int:
        """Cells through the stack: every layer is a whole number of them,
        and the cooling path adds its own on top of the die."""
        return int(round(sum(self.stack_um) / DZ_UM)) + self.cooling_cells

    @property
    def layers(self) -> Tuple[dict, ...]:
        """Every layer, bottom to top, with the cells it occupies. Cooling
        layers report the thickness they REPRESENT, not the cells they use."""
        out, k0 = [], 0
        for name, thickness in zip(LAYER_NAMES, self.stack_um):
            k1 = k0 + int(round(thickness / DZ_UM))
            out.append({"name": name, "k0": k0, "k1": k1, "thickness_um": thickness})
            k0 = k1
        for label, thickness, k_w_mk, cells in self.cooling_layers:
            out.append({"name": label, "k0": k0, "k1": k0 + cells,
                        "thickness_um": thickness, "k_w_mk": k_w_mk, "compact": True})
            k0 += cells
        return tuple(out)

    @property
    def die_z(self) -> Tuple[int, int]:
        n_i = int(round(self.stack_um[0] / DZ_UM))
        n_b = int(round(self.stack_um[1] / DZ_UM))
        return (n_i + n_b, n_i + n_b + self.die_cells)

    @property
    def cell_mm(self) -> float:
        """The finest lateral mesh cell that keeps the model on the direct
        (exact, factorized) solver path. A taller stack spends cells on z, so
        it can force a coarser lateral mesh -- which the app then says."""
        for cell in CELL_CHOICES_MM:
            n = _whole(self.size_mm, cell)
            if n is not None and n >= 8 and n * n * self.nz <= MESH_LEVELS[self.mesh]:
                return cell
        raise StudioError(f"a {self.size_mm:g} mm package does not fit a supported mesh; "
                          f"use a whole number of millimetres")

    @property
    def n_cells(self) -> int:
        return _whole(self.size_mm, self.cell_mm)

    @property
    def keepout_cells(self) -> int:
        """The keep-out in whole mesh cells, rounded UP: a chiplet edge only
        lands on a cell boundary, so any positive keep-out costs at least one
        empty cell between neighbours."""
        return int(math.ceil(self.keepout_um / (self.cell_mm * 1e3) - 1e-9))

    @property
    def keepout_enforced_um(self) -> float:
        """What the mesh can actually enforce -- never less than asked."""
        return self.keepout_cells * self.cell_mm * 1e3

    @property
    def unknowns(self) -> int:
        return self.n_cells * self.n_cells * self.nz

    @property
    def model_key(self) -> Tuple:
        # The mesh level is not in the key: two levels that land on the same
        # cell size are the same model, and should share one factorization.
        return (self.size_mm, self.cell_mm, self.h_w_m2k, self.ambient_c, self.stack_um,
                self.cooling_layers)


@dataclass(frozen=True)
class ChipletIn:
    id: str
    name: str
    w_mm: float
    h_mm: float
    power_w: float
    x_mm: float
    y_mm: float
    power_pattern: str = "uniform"
    blocks: Tuple[PowerBlock, ...] = ()
    power_grid: Optional[PowerGrid] = None

    def as_dict(self) -> dict:
        out = {"id": self.id, "name": self.name, "w_mm": self.w_mm, "h_mm": self.h_mm,
               "power_w": self.power_w, "x_mm": self.x_mm, "y_mm": self.y_mm,
               "power_pattern": self.power_pattern,
               "blocks": [[b.x_frac, b.y_frac, b.w_frac, b.h_frac, b.share] for b in self.blocks]}
        if self.power_grid is not None:
            out["power_grid"] = {"cols": self.power_grid.cols, "rows": self.power_grid.rows,
                                 "weights": list(self.power_grid.weights)}
        return out

    @property
    def hottest_block_density_w_cm2(self) -> float:
        """Power density of the busiest region, which is what a junction
        temperature actually follows -- not the chiplet average."""
        area_cm2 = self.w_mm * self.h_mm / 100.0
        if area_cm2 <= 0:
            return 0.0
        average = self.power_w / area_cm2
        if self.power_grid is not None:
            tiles = self.power_grid.cols * self.power_grid.rows
            total = sum(self.power_grid.weights)
            if total <= 0 or tiles == 0:
                return average
            # Weight of the busiest tile relative to an even share of them.
            return average * max(self.power_grid.weights) / total * tiles
        if not self.blocks:
            return average
        # The background power is spread over the WHOLE footprint, so it flows
        # through the block's area too: the busiest region carries its own
        # share plus that background, which is what block_weights deposits.
        background = uniform_share(self.blocks)
        densest = max((b.share / (b.w_frac * b.h_frac)
                       for b in self.blocks if b.w_frac * b.h_frac > 0), default=0.0)
        return average * (background + densest)


def _parse_power_grid(item: dict, where: str) -> Optional[PowerGrid]:
    """An imported power map: {cols, rows, weights}, relative and row-major
    with the first row at the chiplet's top edge."""
    raw = item.get("power_grid")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise StudioError(f"{where}: power_grid must be an object with cols, rows and weights")
    cols, rows = raw.get("cols"), raw.get("rows")
    for label, value in (("cols", cols), ("rows", rows)):
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_GRID_SIDE:
            raise StudioError(f"{where}: power_grid {label} must be a whole number, "
                              f"1 to {MAX_GRID_SIDE}")
    if cols * rows > MAX_GRID_TILES:
        raise StudioError(f"{where}: a power map may have at most {MAX_GRID_TILES} tiles")
    weights = raw.get("weights")
    if not isinstance(weights, list) or len(weights) != cols * rows:
        raise StudioError(f"{where}: power_grid needs {cols * rows} weights "
                          f"({cols} x {rows}), got {len(weights) if isinstance(weights, list) else 'none'}")
    values = []
    for n, value in enumerate(weights, start=1):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise StudioError(f"{where}: power_grid value {n} is not a number")
        if value < 0:
            raise StudioError(f"{where}: power_grid value {n} is negative; "
                              f"a region cannot consume less than no power")
        values.append(float(value))
    if sum(values) <= 0:
        raise StudioError(f"{where}: the power map is empty -- every tile is zero")
    return PowerGrid(cols=cols, rows=rows, weights=tuple(values))


def _parse_blocks(item: dict, where: str) -> Tuple[str, Tuple[PowerBlock, ...]]:
    """The chiplet's internal power map: a named pattern, or explicit blocks.

    Explicit blocks win when both are given, so a design file can carry a map
    the presets cannot express.
    """
    pattern = item.get("power_pattern", "uniform")
    # "custom" and "imported" are what the engine ITSELF reports for a chiplet
    # whose map came from blocks or a file, and the app sends that back on the
    # next request. Refusing them would break the round trip: the blocks or
    # the grid in the same object are what actually define the heat.
    if not isinstance(pattern, str) or (pattern not in POWER_PATTERNS
                                        and pattern not in ("custom", "imported")):
        raise StudioError(f"{where}: power_pattern must be one of "
                          f"{', '.join(sorted(POWER_PATTERNS))}")
    if pattern in ("custom", "imported") and item.get("blocks") is None and item.get("power_grid") is None:
        pattern = "uniform"        # nothing to describe; it is a plain chiplet
    raw_blocks = item.get("blocks")
    if raw_blocks is None:
        return pattern, tuple(PowerBlock(*values) for values in POWER_PATTERNS.get(pattern, ()))
    if not isinstance(raw_blocks, list):
        raise StudioError(f"{where}: blocks must be a list")
    if len(raw_blocks) > MAX_BLOCKS:
        raise StudioError(f"{where}: at most {MAX_BLOCKS} internal blocks")
    blocks: List[PowerBlock] = []
    total = 0.0
    for n, values in enumerate(raw_blocks, start=1):
        if not isinstance(values, (list, tuple)) or len(values) != 5:
            raise StudioError(f"{where}: block {n} must be [x, y, width, height, share] "
                              f"as fractions of the footprint")
        x, y, w, h, share = (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool)
                             and math.isfinite(v) else None for v in values)
        if None in (x, y, w, h, share):
            raise StudioError(f"{where}: block {n} must be five numbers")
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            raise StudioError(f"{where}: block {n} must sit inside the footprint")
        if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
            raise StudioError(f"{where}: block {n} runs past the edge of the chiplet")
        if not 0.0 <= share <= 1.0:
            raise StudioError(f"{where}: block {n} share must be between 0 and 1")
        total += share
        blocks.append(PowerBlock(x, y, w, h, share))
    if total > 1.0 + 1e-9:
        raise StudioError(f"{where}: internal blocks claim {total:.2f} of the power, "
                          f"which is more than the chiplet has")
    return (pattern if not blocks else item.get("power_pattern", "custom")), tuple(blocks)


def parse_chiplets(raw, spec: PackageSpec) -> Tuple[ChipletIn, ...]:
    if not isinstance(raw, list) or not raw:
        raise StudioError("add at least one chiplet")
    if len(raw) > MAX_CHIPLETS:
        raise StudioError(f"at most {MAX_CHIPLETS} chiplets")
    cell = spec.cell_mm
    seen = set()
    out: List[ChipletIn] = []
    for n, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise StudioError(f"chiplet {n} must be an object")
        cid = item.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise StudioError(f"chiplet {n}: id is missing")
        if cid in seen:
            raise StudioError(f"duplicate chiplet id '{cid}'")
        seen.add(cid)
        name = str(item.get("name") or cid)[:40]
        pattern, blocks = _parse_blocks(item, name)
        power_grid = _parse_power_grid(item, name)
        if power_grid is not None:
            pattern, blocks = "imported", ()      # a measured map replaces the presets
        chiplet = ChipletIn(
            id=cid, name=name,
            w_mm=_number(item, "w_mm", cell, spec.size_mm, name),
            h_mm=_number(item, "h_mm", cell, spec.size_mm, name),
            power_w=_number(item, "power_w", 0.0, 2000.0, name),
            x_mm=_number(item, "x_mm", 0.0, spec.size_mm, name),
            y_mm=_number(item, "y_mm", 0.0, spec.size_mm, name),
            power_pattern=pattern, blocks=blocks, power_grid=power_grid,
        )
        for label, value in (("width", chiplet.w_mm), ("height", chiplet.h_mm),
                             ("x", chiplet.x_mm), ("y", chiplet.y_mm)):
            if _whole(value, cell) is None:
                raise StudioError(f"{name}: {label} {value:g} mm is not on the {cell:g} mm grid")
        out.append(chiplet)
    return tuple(out)


def _catalog(chiplets) -> Tuple[ChipletDef, ...]:
    return tuple(ChipletDef(id=c.id, name=c.name, width_um=c.w_mm * 1e3,
                            height_um=c.h_mm * 1e3, power_w=c.power_w,
                            blocks=c.blocks, power_grid=c.power_grid) for c in chiplets)


def _placement(chiplets, pitch_mm: float) -> Placement:
    return Placement(pitch_m=pitch_mm * 1e-3,
                     origins_m=tuple((c.id, c.x_mm * 1e-3, c.y_mm * 1e-3) for c in chiplets))


# ---------------------------------------------------------------------------
# The factorized package model, cached
# ---------------------------------------------------------------------------

class SymmetricFactor:
    """Solves the solver's system M x = b through its symmetric form.

    S = -diag(rho_cp) * M is symmetric positive definite, so it can be
    factorized with a symmetric fill-reducing ordering. Measured on the
    40x40x12 model: 5.4 M factor nonzeros instead of 10.9 M, a 1.4 s
    factorization instead of ~4 s, triangular solves twice as fast, and
    agreement with the solver's own factorization to 5e-12 relative.

    Installed as the solver's cached factorization, so every solve still goes
    through ThermalSolver.solve_steady_state() and its residual, energy-balance
    and convective-floor checks.
    """

    def __init__(self, solver: ThermalSolver):
        n = solver.nx * solver.ny * solver.nz
        self.rho_cp = solver._rho_cp.reshape(n)
        spd = (-(sparse.diags(self.rho_cp) @ solver._conduction_operator)).tocsc()
        spd = ((spd + spd.T) * 0.5).tocsc()
        # Kept so a batch of solves can be checked against the system it
        # claims to have solved, rather than against one sample of itself.
        self._spd = spd.tocsr()
        self._lu = sparse_linalg.splu(spd, permc_spec="MMD_AT_PLUS_A",
                                      diag_pivot_thresh=0.0,
                                      options={"SymmetricMode": True})

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """x with M x = rhs (the interface solve_steady_state() expects)."""
        rhs = np.asarray(rhs, dtype=np.float64)
        scale = self.rho_cp[:, None] if rhs.ndim == 2 else self.rho_cp
        return self._lu.solve(-(scale * rhs))

    def rise_for_power(self, q_w_m3: np.ndarray, workers: int = 1) -> np.ndarray:
        """Temperature rise theta with S theta = q, columns of q in W/m^3.

        A batch of columns is split across threads when `workers` > 1: the
        factorization is read-only and SuperLU releases the GIL, so this
        scales (measured 2x on a 28,800-cell model). Because that object is
        not documented as thread-safe, the whole batch is then substituted
        back into S theta = q; a batch that does not satisfy the system it
        solved is refused rather than reported.
        """
        q = np.asarray(q_w_m3, dtype=np.float64)
        if q.ndim != 2 or workers <= 1 or q.shape[1] < 2 * SOLVE_CHUNK:
            return self._lu.solve(q)

        bounds = list(range(0, q.shape[1], SOLVE_CHUNK))
        chunks = [slice(lo, min(lo + SOLVE_CHUNK, q.shape[1])) for lo in bounds]
        with ThreadPoolExecutor(max_workers=min(workers, len(chunks)),
                                thread_name_prefix="risc3d-solve") as pool:
            parts = list(pool.map(lambda c: self._lu.solve(np.asfortranarray(q[:, c])), chunks))
        out = np.concatenate(parts, axis=1)

        # Substitution, not a sample: every column must satisfy S theta = q.
        scale = max(1.0, float(np.max(np.abs(q))))
        residual = float(np.max(np.abs(self._spd @ out - q)))
        if residual > THREADED_SOLVE_TOL * scale:
            raise StudioError(
                f"a threaded batch of solves does not satisfy the system it solved "
                f"(residual {residual:.2e} against a limit of {THREADED_SOLVE_TOL * scale:.2e}); "
                f"refusing to use it")
        return out

def _apply_cooling_materials(grid_builder, spec: PackageSpec) -> None:
    """Writes the compact cooling layers over the cells build_stack filled
    with silicon. See COOLING_STACKS for what "compact" means and why.
    """
    if not spec.cooling_layers:
        return
    k0 = spec.die_z[1]
    for _label, thickness_um, k_w_mk, cells in spec.cooling_layers:
        t_model_um = cells * DZ_UM
        ratio = thickness_um / t_model_um
        k_inplane = k_w_mk * ratio          # same sheet conductance k * t
        k_through = k_w_mk / ratio          # same through-plane resistance t / k
        zs = slice(k0, k0 + cells)
        grid_builder.kx_map[:, :, zs] = k_inplane
        grid_builder.ky_map[:, :, zs] = k_inplane
        grid_builder.kz_map[:, :, zs] = k_through
        # Thermal mass scaled the same way, so a transient mode would see the
        # real heat capacity of the layer rather than the modelled slab's.
        grid_builder.rho_map[:, :, zs] = grid_builder.rho_map[:, :, zs] * ratio
        k0 += cells


class CrankNicolsonFactor:
    """Solves the implicit time step through its symmetric form.

    A Crank-Nicolson step solves (I - dt/2 M) x = b. Multiplying by
    D = diag(rho_cp) turns that into (D + dt/2 S) x = D b, where S = -D M is
    the same symmetric positive-definite operator the steady path factorizes
    -- so the step gets a symmetric, fill-reducing factorization instead of a
    general one. Installed in the solver's own cache, so time stepping still
    runs through ThermalSolver.step_implicit and its finite-field check.

    Without this, a mesh above the protected solver's direct-path threshold
    falls to Jacobi-preconditioned BiCGSTAB: measured 732 ms per step against
    34 ms, which is the difference between a demo and a coffee break.
    """

    def __init__(self, factor: "SymmetricFactor", dt: float):
        system = (sparse.diags(factor.rho_cp) + (dt / 2.0) * factor._spd).tocsc()
        system = ((system + system.T) * 0.5).tocsc()
        self._lu = sparse_linalg.splu(system, permc_spec="MMD_AT_PLUS_A",
                                      diag_pivot_thresh=0.0, options={"SymmetricMode": True})
        self.rho_cp = factor.rho_cp

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._lu.solve(self.rho_cp * np.asarray(rhs, dtype=np.float64))


class ThermalModel:
    """One package: materials, boundary conditions and the factorized
    operator. Built once per (size, cell, h, ambient); every placement after
    that costs one triangular solve."""

    def __init__(self, spec: PackageSpec):
        t0 = time.perf_counter()
        n, cell_m = spec.n_cells, spec.cell_mm * 1e-3
        # The cooling path rides on top of the die. The stack section only
        # knows three layers, so the mesh is built with a die tall enough to
        # hold the cooling cells and their materials are written afterwards;
        # the Package is then corrected so "the die" means the real die.
        die_um_meshed = spec.stack_um[2] + spec.cooling_cells * DZ_UM
        config = PipelineConfig(
            grid=GridSection(nx=n, ny=n, nz=spec.nz, dx=cell_m, dy=cell_m, dz=DZ_M),
            boundary_conditions=BoundaryConditionsSection(
                h_convective=spec.h_w_m2k, t_ambient_c=spec.ambient_c,
                bottom_bc="neumann", top_bc="robin", side_bc="neumann"),
            stack=StackSection(interposer_um=spec.stack_um[0], bump_um=spec.stack_um[1],
                               die_um=die_um_meshed),
            power=PowerSection(die_power_w=0.0, die_footprint_frac=0.5),
            solver=SolverSection(), visualization=VisualizationSection(),
            case=CaseSection(name="studio"),
        )
        config.validate()
        self.spec = spec
        meshed = package_from_config(config)
        self.package = replace(
            meshed, die_z=spec.die_z,
            layer_cells=(meshed.layer_cells[0], meshed.layer_cells[1], spec.die_cells),
            stack_um=spec.stack_um)
        grid_builder = build_stack(config)
        _apply_cooling_materials(grid_builder, spec)
        grid_builder.validate_grid()
        self.solver = ThermalSolver.from_grid_builder(grid_builder)
        self.factor = SymmetricFactor(self.solver)
        self.solver._steady_lu = self.factor                    # used by every solve
        self.solver.solve_steady_state(apply_result=False)      # verifies the factor
        n_cells = self.package.nx * self.package.ny * self.package.nz
        z0, z1 = power_z_range(self.package)
        self.power_z = (z0, z1)
        flat = np.arange(n_cells).reshape(self.package.nx, self.package.ny, self.package.nz)
        # The cells that carry power -- the device layer at the bumped face of
        # the die. By the discrete maximum principle the peak of any layout
        # lies in a powered cell (every source is non-negative and the sink can
        # only cool), so the search needs only these; the full re-solve, which
        # takes its peak over the WHOLE field, checks that on every run.
        # Layer-major, bottom layer first: interior bounds use the leading
        # layers, complete layouts are scored over all of them.
        self.die_cells = np.ascontiguousarray(
            np.concatenate([flat[:, :, k].reshape(-1) for k in range(z0, z1)]))
        self.bound_cells = int(min(3, z1 - z0) * self.package.nx * self.package.ny)
        self.lambda_mm = spreading_length_m(self.package) * 1e3
        self.lock = threading.Lock()
        self.build_s = time.perf_counter() - t0


class ModelCache:
    """Built models, kept while they are cheap to keep.

    Eviction is by SIZE as well as count: a 0.25 mm model carries a
    factorization of about 640 MB, so holding six of them would cost four
    gigabytes and push the machine into paging -- which is exactly how a
    fast app becomes a slow one without any single operation looking slow.
    """

    def __init__(self, capacity: int = 6, max_cells: int = MAX_CACHED_CELLS):
        self._models: "OrderedDict[tuple, ThermalModel]" = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = capacity
        self.max_cells = max_cells

    def _cells(self) -> int:
        return sum(m.package.nx * m.package.ny * m.package.nz for m in self._models.values())

    def get(self, spec: PackageSpec) -> Tuple[ThermalModel, bool]:
        with self._lock:
            key = spec.model_key
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key], False
            model = ThermalModel(spec)
            self._models[key] = model
            while len(self._models) > self.capacity or (
                    len(self._models) > 1 and self._cells() > self.max_cells):
                self._models.popitem(last=False)
            return model, True


# ---------------------------------------------------------------------------
# One placement
# ---------------------------------------------------------------------------

def _load_study(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_MESH_STUDY = _load_study(MESH_STUDY_PATH)
_MATERIAL_STUDY = _load_study(MATERIAL_STUDY_PATH)
_BOUNDARY_STUDY = _load_study(BOUNDARY_STUDY_PATH)


# A source needs about this many cells across it before its PEAK is the
# design's rather than the mesh's. Below one cell the finite-volume average is
# all there is: the same cell carries the same watts whether the hotspot
# inside it is 0.5 mm or 0.1 mm, so the reported peak stops depending on the
# design. Two cells is where measured refinement stops moving the answer by
# more than the band: at 2 cells across, 0.5 mm -> 0.25 mm moved the peak
# 1.04 K against a 1.98 K band; at 1 cell it moved 29.2 K.
MIN_CELLS_ACROSS_FEATURE = 2.0


def smallest_feature_mm(chiplet) -> Optional[float]:
    """The narrowest thing inside this chiplet, in mm, or None if uniform.

    For blocks that is the smaller side of the smallest block; for an imported
    map it is one tile, since a tile is the finest structure the file asserts.
    """
    sizes = []
    for b in chiplet.blocks:
        sizes.append(min(b.w_frac * chiplet.w_mm, b.h_frac * chiplet.h_mm))
    grid = chiplet.power_grid
    if grid is not None and grid.cols and grid.rows:
        sizes.append(min(chiplet.w_mm / grid.cols, chiplet.h_mm / grid.rows))
    return min(sizes) if sizes else None


def resolution_report(spec: PackageSpec, chiplets) -> dict:
    """Is the hottest structure in this design resolved by the mesh?

    This is not the discretization band, and the band does not cover it. The
    band says how much the answer moves when the mesh is refined around a
    source the mesh can SEE. A source smaller than a cell is not seen at all:
    its power lands in one cell and is averaged over the cell's area, so the
    reported peak is a LOWER BOUND on the junction, and refining raises it
    instead of converging.

    Measured on a 5 mm part with a centred hotspot carrying 70 % of 8 W, on a
    20 mm package at h = 2000:

        hotspot 3.15 mm (6 cells)   0.5 mm 81.8 C   0.25 mm 82.0 C    +0.2 K
        hotspot 1.00 mm (2 cells)   0.5 mm 117.3 C  0.25 mm 118.3 C   +1.0 K
        hotspot 0.50 mm (1 cell)    0.5 mm 117.3 C  0.25 mm 146.5 C  +29.2 K
        hotspot 0.10 mm             0.5 mm 117.3 C  0.25 mm 146.5 C  +29.2 K

    The last two rows are the point: identical answers for hotspots that
    differ by 5x, because both fill one cell. The app cannot tell them apart,
    so it has to say so rather than print a confident number.
    """
    worst_id, worst_mm = None, None
    for c in chiplets:
        feature = smallest_feature_mm(c)
        if feature is not None and (worst_mm is None or feature < worst_mm):
            worst_id, worst_mm = c.id, feature
    if worst_mm is None:
        return {"applies": False, "resolved": True, "cell_mm": spec.cell_mm}
    cells_across = worst_mm / spec.cell_mm
    return {
        "applies": True,
        "resolved": bool(cells_across >= MIN_CELLS_ACROSS_FEATURE),
        "under_one_cell": bool(cells_across < 1.0),
        "chiplet": worst_id,
        "feature_mm": worst_mm,
        "cell_mm": spec.cell_mm,
        "cells_across": cells_across,
        "needed_cell_mm": worst_mm / MIN_CELLS_ACROSS_FEATURE,
        "reachable": bool(worst_mm / MIN_CELLS_ACROSS_FEATURE >= min(CELL_CHOICES_MM)),
    }


def mesh_check(cache: "ModelCache", spec: PackageSpec, chiplets) -> dict:
    """Solve THIS design at both mesh levels and report the difference.

    The band on every other result is an estimate: it is measured on reference
    packages and scaled, so it describes designs like them rather than the one
    on screen. This does not estimate anything. It solves the same design at
    0.5 mm and again at 0.25 mm and subtracts, which is the discretization
    error of this design, measured.

    It costs seconds rather than milliseconds, which is why it is a thing the
    user asks for rather than something every solve does. What it is for: a
    design close enough to its limit that the estimate is the difference
    between passing and failing.
    """
    levels: Dict[str, dict] = {}
    for level in MESH_LEVELS:
        at = dataclasses.replace(spec, mesh=level)
        started = time.perf_counter()
        out = solve(cache, at, chiplets)
        if not out["ok"]:
            raise StudioError("the design must be valid before its mesh error can be "
                              f"measured: {'; '.join(out['violations'])}")
        levels[level] = {
            "mesh": level,
            "cell_mm": at.cell_mm,
            "unknowns": at.unknowns,
            "peak_c": out["peak"]["t_c"],
            "residual": out["verification"]["residual"],
            "estimated_band_k": (out["metrics"]["uncertainty"] or {}).get("mesh_k"),
            "solve_s": time.perf_counter() - started,
        }
    ordered = sorted(levels.values(), key=lambda row: -row["cell_mm"])
    coarse, fine = ordered[0], ordered[-1]
    measured = abs(fine["peak_c"] - coarse["peak_c"])
    estimated = coarse["estimated_band_k"]
    same_mesh = coarse["cell_mm"] == fine["cell_mm"]
    return {
        "ok": True,
        "levels": ordered,
        "same_mesh": same_mesh,
        "measured_k": None if same_mesh else measured,
        "estimated_k": estimated,
        "covered": None if same_mesh or estimated is None else bool(measured <= estimated),
        "use_peak_c": fine["peak_c"],
        "note": ("this package already meshes at the finest cell the app builds, so "
                 "both levels are the same solve and there is nothing to compare"
                 if same_mesh else
                 "measured on this design: the difference between solving it at "
                 f"{coarse['cell_mm']:g} mm and at {fine['cell_mm']:g} mm"),
    }


def any_internal_power(chiplets) -> bool:
    """Does any part carry structure inside its footprint?

    Blocks and imported maps both put power somewhere other than everywhere,
    and both are resolved by the mesh rather than by the chiplet outline. That
    is what decides which measured band applies.
    """
    return any(c.blocks or c.power_grid is not None for c in chiplets)


def _case_for(h_w_m2k: float, cases: dict) -> Optional[str]:
    """Which measured case this design resembles. Harder cooling means
    steeper gradients, and both studies were run for both regimes."""
    if not cases:
        return None
    name = "coldplate" if h_w_m2k >= 10000.0 else "passive"
    return name if name in cases else next(iter(cases))


def error_budget(cell_mm: float, rise_k: float, h_w_m2k: float,
                 mesh_study: Optional[dict] = None,
                 material_study: Optional[dict] = None,
                 concentrated: bool = False,
                 resolution: Optional[dict] = None,
                 boundary_study: Optional[dict] = None) -> Optional[dict]:
    """What is known about how wrong the peak could be, in kelvin.

    Three measured, independent contributions, added in quadrature:
      mesh      -- reference packages refined to 0.25 mm and compared
                   (tools/mesh_convergence.py), audited by tools/band_check.py
      materials -- silicon conductivity +/-10 %, bump composite +/-30 %
                   (tools/material_sensitivity.py)
      boundary  -- the convective coefficient +/-9 %, which is how much a real
                   rig's own h varies across 13 power patterns
                   (tools/boundary_sensitivity.py)

    The boundary term is the largest of the three, at about 5 % of the rise
    against 1-2 % for the mesh. That is worth saying plainly: the number is
    limited by how well the cooling is known, not by the solver.

    Both are measured on a reference package and scaled to THIS design's rise
    above ambient, which is what they are proportional to. This is a
    sensitivity budget, not a validation bound: it says how much the answer
    moves when the model's own inputs move, and nothing about whether the
    model matches hardware. Returns None when the studies have not been run.
    """
    mesh = mesh_band(cell_mm, rise_k, h_w_m2k, mesh_study, concentrated)
    material_study = _MATERIAL_STUDY if material_study is None else material_study
    material_k = None
    if material_study and rise_k > 0:
        cases = material_study.get("cases") or {}
        name = _case_for(h_w_m2k, cases)
        case = cases.get(name) if name else None
        if case and case.get("rise_k"):
            material_k = float(case["material_band_k"]) / float(case["rise_k"]) * rise_k
    boundary_study = _BOUNDARY_STUDY if boundary_study is None else boundary_study
    boundary_k = None
    if boundary_study and rise_k > 0:
        cases = (boundary_study.get("cases") or {}).values()
        pcts = [float(c["boundary_band_pct_of_rise"]) for c in cases
                if c.get("boundary_band_pct_of_rise")]
        # The worst of the measured regimes, for the same reason the mesh band
        # takes its worst case: a band that holds beats one that flatters.
        if pcts:
            boundary_k = max(pcts) / 100.0 * rise_k
    if mesh is None and material_k is None and boundary_k is None:
        return None
    mesh_k = mesh["band_k"] if mesh else 0.0
    combined = math.sqrt(mesh_k ** 2 + (material_k or 0.0) ** 2 + (boundary_k or 0.0) ** 2)
    # A band describes a source the mesh can see. If the design's smallest
    # feature is under a cell, refining does not converge the peak, it raises
    # it -- so quoting a band here would be quoting the wrong quantity.
    limited = bool(resolution and resolution.get("applies") and not resolution.get("resolved"))
    return {
        "mesh_k": mesh_k,
        "material_k": material_k,
        "boundary_k": boundary_k,
        "resolution_limited": limited,
        "peak_is_lower_bound": limited,
        "combined_k": combined,
        "pct_of_rise": 100.0 * combined / rise_k if rise_k else 0.0,
        "cell_mm": mesh["cell_mm"] if mesh else cell_mm,
        "measured_on": mesh["measured_on"] if mesh else _case_for(h_w_m2k, (material_study or {}).get("cases") or {}),
        "observed_order": mesh["observed_order"] if mesh else None,
        "concentrated": concentrated,
    }


def mesh_band(cell_mm: float, rise_k: float, h_w_m2k: float,
              study: Optional[dict] = None, concentrated: bool = False) -> Optional[dict]:
    """How much the peak still moves with the mesh, in kelvin.

    Measured by refining reference packages to 0.25 mm and comparing, then
    scaled to THIS design's rise above ambient. It is a discretization band,
    not a validation bound: it says how much of the answer is the mesh, and
    nothing about whether the model matches hardware.

    Two things pick the band, in this order.

    FIRST, whether this design has internal power patterns, because that is
    what dominates the error. A chiplet outline lands on a cell boundary; the
    blocks inside it are fractions of the footprint and do not. Measured on
    the shipped RISC-V scenario, 0.5 mm against 0.25 mm:

        every part uniform               0.004 K
        the same design, MAC-array NPU   1.008 K
        that NPU alone on a bare die     2.693 K

    So the uniform reference cases understate a patterned design by a factor
    of twenty, and patterns are exactly the feature that makes the answer
    realistic. The study measures both families (tools/mesh_convergence.py)
    and this picks the family the design belongs to.

    SECOND, within that family, the WORST case rather than the one whose
    cooling matches -- error tracks how concentrated the sources are, not only
    the rise, and a band that flatters is worse than no band because somebody
    spends it.

    Both rules were forced by measurement, not chosen. tools/band_check.py
    solves fourteen designs at both mesh levels and compares what it sees
    against what this function quoted them; four exceeded their band, the
    worst by 2.85x, until the study gained reference cases as concentrated as
    the designs the app permits. All fourteen now come in under, the closest
    at 0.81 of its band.

    It remains an ESTIMATE from reference packages, and it is conservative for
    most designs because it has to hold for the worst. The design in hand can
    be measured exactly instead -- see mesh_check(). Returns None when no
    study has been run.
    """
    study = _MESH_STUDY if study is None else study
    if not study or rise_k <= 0:
        return None
    cases = study.get("cases") or {}
    # The family this design belongs to. A study written before the patterned
    # cases existed has none of them, so an unlabelled case counts as uniform
    # and a patterned design falls back to every case there is.
    family = {n: c for n, c in cases.items() if bool(c.get("concentrated")) == concentrated}
    if not family:
        family = cases
    key, pct, worst = None, 0.0, None
    for name, case in family.items():
        table = case.get("band_pct_of_rise_by_cell_mm") or {}
        if not table:
            continue
        cell_key = f"{cell_mm:g}"
        if cell_key not in table:
            # Fall back to the coarsest measured mesh, which bands no tighter.
            cell_key = max(table, key=lambda k: float(k))
        value = float(table[cell_key])
        if value > pct:
            key, pct, worst = cell_key, value, name
    if key is None:
        return None
    return {
        "band_k": pct / 100.0 * rise_k,
        "band_pct_of_rise": pct,
        "cell_mm": float(key),
        "measured_on": (f"worst of {len(family)} reference cases ({worst})"
                        + (" with internal power patterns" if concentrated
                           else " with uniform chiplet power")),
        "observed_order": (family.get(worst) or {}).get("order_p"),
        "concentrated": concentrated,
        "estimated": True,
    }


def result_payload(model: ThermalModel, chiplets, result) -> dict:
    """Everything the UI shows for one solved placement."""
    spec, pkg = model.spec, model.package
    field_c = np.asarray(result.field_k, dtype=np.float64) - 273.15
    total_w = float(sum(c.power_w for c in chiplets))
    area_mm2 = spec.size_mm ** 2
    chip_mm2 = float(sum(c.w_mm * c.h_mm for c in chiplets))
    floor_k = total_w / (spec.h_w_m2k * area_mm2 * 1e-6)
    rise_k = float(result.peak_c) - spec.ambient_c
    i, j, k = result.peak_ijk
    layer = next(l["name"] for l in spec.layers if l["k0"] <= k < l["k1"])
    resolution = resolution_report(spec, chiplets)
    by_id = {c.id: c for c in chiplets}
    rows = []
    for row in result.per_chiplet:
        c = by_id[row.id]
        area_cm2 = c.w_mm * c.h_mm / 100.0
        rows.append({"id": c.id, "name": c.name, "max_c": float(row.max_c),
                     "mean_c": float(row.mean_c), "power_w": c.power_w,
                     "density_w_cm2": c.power_w / area_cm2 if area_cm2 > 0 else 0.0,
                     "power_pattern": c.power_pattern,
                     "hottest_density_w_cm2": c.hottest_block_density_w_cm2})
    return {
        "grid": {"nx": pkg.nx, "ny": pkg.ny, "nz": pkg.nz, "cell_mm": spec.cell_mm,
                 "dz_um": pkg.dz * 1e6, "size_mm": spec.size_mm,
                 "layers": [dict(layer_) for layer_ in spec.layers],
                 "stack_um": list(spec.stack_um),
                 "die_k0": spec.die_z[0], "die_k1": spec.die_z[1],
                 "mesh": spec.mesh, "unknowns": spec.unknowns,
                 "cooling": spec.cooling,
                 "cooling_layers": [list(layer_) for layer_ in spec.cooling_layers]},
        "field_b64": base64.b64encode(field_c.astype("<f4").tobytes(order="C")).decode("ascii"),
        "t_min_c": float(field_c.min()),
        "t_max_c": float(field_c.max()),
        "ambient_c": spec.ambient_c,
        "resolution": resolution,
        "peak": {"t_c": float(result.peak_c), "i": i, "j": j, "k": k,
                 "x_mm": (i + 0.5) * spec.cell_mm, "y_mm": (j + 0.5) * spec.cell_mm,
                 "z_um": (k + 0.5) * pkg.dz * 1e6, "layer": layer},
        "chiplets": rows,
        "hottest_chiplet": max(rows, key=lambda r: r["max_c"])["id"] if rows else None,
        "metrics": {
            "total_power_w": total_w, "rise_k": rise_k,
            "rja_k_per_w": rise_k / total_w if total_w > 0 else None,
            "floor_rise_k": floor_k,
            "floor_ratio": rise_k / floor_k if floor_k > 0 else None,
            "package_area_mm2": area_mm2, "chiplet_area_mm2": chip_mm2,
            "fill_pct": 100.0 * chip_mm2 / area_mm2,
            "flux_package_w_cm2": total_w / (area_mm2 / 100.0),
            "flux_chiplets_w_cm2": total_w / (chip_mm2 / 100.0) if chip_mm2 > 0 else None,
            "lambda_mm": model.lambda_mm, "h_w_m2k": spec.h_w_m2k,
            "uncertainty": error_budget(spec.cell_mm, rise_k, spec.h_w_m2k,
                                        concentrated=any_internal_power(chiplets),
                                        resolution=resolution),
        },
        "verification": {
            "residual": float(result.residual),
            "energy_imbalance_k": float(result.energy_imbalance_k),
            "sink_heat_w": float(result.sink_heat_w),
            "deposited_w": float(result.power.deposited_w),
            "specified_w": total_w,
            "unknowns": int(pkg.nx * pkg.ny * pkg.nz),
        },
        "solve_ms": float(result.solve_s * 1e3),
    }


def solve(cache: ModelCache, spec: PackageSpec, chiplets) -> dict:
    """Solve one placement. Invalid placements are reported, never solved."""
    model, built = cache.get(spec)
    catalog = _catalog(chiplets)
    placement = _placement(chiplets, spec.cell_mm)
    violations = validate_placement(model.package, catalog, placement,
                                    spec.keepout_enforced_um * 1e-6)
    if violations:
        return {"ok": False, "violations": violations}
    with model.lock:
        result = solve_placement(model.solver, model.package, catalog, placement, "placement")
    payload = result_payload(model, chiplets, result)
    payload["ok"] = True
    payload["model_built_s"] = model.build_s if built else 0.0
    return payload


# ---------------------------------------------------------------------------
# The exact optimizer
# ---------------------------------------------------------------------------

MIN_TRANSIENT_DT_S = 1e-3
MAX_TRANSIENT_DT_S = 0.05
MAX_TRANSIENT_STEPS = 600


def parse_profile(raw) -> dict:
    """What the chiplets do over time: idle, then a burst, then idle again."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise StudioError("transient must be an object")
    profile = {
        "idle_fraction": _number({"v": raw.get("idle_fraction", 0.2)}, "v", 0.0, 1.0, "transient"),
        "burst_s": _number({"v": raw.get("burst_s", 1.0)}, "v", 0.05, 60.0, "transient"),
        "cool_s": _number({"v": raw.get("cool_s", 1.0)}, "v", 0.0, 60.0, "transient"),
    }
    limit = raw.get("limit_c")
    profile["limit_c"] = (None if limit is None
                          else _number({"v": limit}, "v", -40.0, 500.0, "transient"))
    return profile


def transient(cache: ModelCache, spec: PackageSpec, chiplets, profile: dict,
              cancel: Optional[threading.Event] = None,
              on_progress: Optional[Callable[[dict], None]] = None) -> dict:
    """How the package behaves over time, not just where it settles.

    The die starts at the steady state of its idle power, the chiplets go to
    full power for `burst_s`, and then drop back. Every step is a
    Crank-Nicolson step of the same verified solver; the thermal mass that
    sets the timescale is the same rho*cp the materials carry, including the
    compact cooling layers, whose density is scaled so their heat capacity is
    the real layer's.

    The check that this is the same physics as the steady answer: run long
    enough and the transient MUST arrive at the steady field. That comparison
    is reported with the result rather than assumed.
    """
    model, _built = cache.get(spec)
    catalog = _catalog(chiplets)
    placement = _placement(chiplets, spec.cell_mm)
    violations = validate_placement(model.package, catalog, placement,
                                    spec.keepout_enforced_um * 1e-6)
    if violations:
        return {"ok": False, "violations": violations}

    pkg = model.package
    idle = profile["idle_fraction"]
    t_start = time.perf_counter()

    with model.lock:
        solver = model.solver
        q_full = np.asarray(build_power_map(pkg, catalog, placement).q_w_m3, dtype=np.float64)

        # The timescale, from the heat the package can store against the rate
        # it can shed: tau = (sum rho*cp*volume) / (h*A). It sets the step and
        # the length of the run, so neither is a guess.
        capacity_j_k = float(np.sum(solver._rho_cp)) * pkg.cell_volume_m3
        conductance_w_k = spec.h_w_m2k * pkg.sink_area_m2
        tau_s = capacity_j_k / conductance_w_k if conductance_w_k > 0 else 1.0
        dt = min(MAX_TRANSIENT_DT_S, max(MIN_TRANSIENT_DT_S, tau_s / 40.0))
        total_s = profile["burst_s"] + profile["cool_s"]
        steps = int(math.ceil(total_s / dt))
        if steps > MAX_TRANSIENT_STEPS:
            dt = total_s / MAX_TRANSIENT_STEPS
            steps = MAX_TRANSIENT_STEPS

        # Idle steady state as the starting field.
        solver.q_map[:] = q_full * idle
        solver.solve_steady_state(apply_result=True)
        start_peak_c = float(solver.T.max()) - 273.15

        # The step's own factorization, symmetric like the steady one.
        solver._cn_solver_cache.clear()
        identity = sparse.identity(pkg.nx * pkg.ny * pkg.nz, format="csr")
        lhs = (identity - (dt / 2.0) * solver._conduction_operator).tocsc()
        solver._cn_solver_cache[dt] = {
            "backend": "direct", "lu": CrankNicolsonFactor(model.factor, dt), "lhs": lhs,
        }

        die0, die1 = pkg.die_z
        rows = []
        for c in chiplets:
            i0, j0 = int(round(c.x_mm / spec.cell_mm)), int(round(c.y_mm / spec.cell_mm))
            nvx, nvy = int(round(c.w_mm / spec.cell_mm)), int(round(c.h_mm / spec.cell_mm))
            rows.append((c.id, i0, j0, nvx, nvy))

        times = [0.0]
        peaks = [start_peak_c]
        per_chiplet = {cid: [float(solver.T[i0:i0 + nvx, j0:j0 + nvy, die0:die1].max()) - 273.15]
                       for cid, i0, j0, nvx, nvy in rows}
        limit = profile["limit_c"]
        time_to_limit = 0.0 if (limit is not None and start_peak_c >= limit) else None

        burst_steps = int(round(profile["burst_s"] / dt))
        solver.q_map[:] = q_full
        for step in range(1, steps + 1):
            _check_cancel(cancel)
            if step == burst_steps + 1:
                solver.q_map[:] = q_full * idle          # the burst ends
            solver.step_implicit(dt)
            t = step * dt
            peak = float(solver.T.max()) - 273.15
            times.append(t)
            peaks.append(peak)
            for cid, i0, j0, nvx, nvy in rows:
                per_chiplet[cid].append(
                    float(solver.T[i0:i0 + nvx, j0:j0 + nvy, die0:die1].max()) - 273.15)
            if time_to_limit is None and limit is not None and peak >= limit:
                time_to_limit = t
            if on_progress is not None and step % 10 == 0:
                on_progress({"stage": "stepping", "step": step, "steps": steps,
                             "t_s": t, "peak_c": peak})

        # The claim this rests on: keep going and it arrives at the steady
        # answer. Held to the full-power field, from this same solver.
        solver.q_map[:] = q_full
        steady = solver.solve_steady_state(apply_result=False)
        steady_peak_c = float(steady.max()) - 273.15
        burst_peak_c = max(peaks[:burst_steps + 1]) if burst_steps else start_peak_c

        # The step's factorization is scratch for THIS run: it is a second
        # factorization of a package-sized system, hundreds of megabytes on a
        # meshed cooling path, and the model it hangs off is cached and shared.
        # Keeping it would leak that much per burst -- measured 20 to 80 MB a
        # run before this line existed, on top of an already cached model.
        solver._cn_solver_cache.clear()
        solver.T[:] = pkg.t_ambient_k
        solver.time_s = 0.0

    # Crank-Nicolson is unconditionally stable but not monotone: a step
    # change in power at a dt far above the explicit CFL reference makes the
    # hottest point wobble by a little before it settles. That is a known
    # artefact of the scheme, not physics, so it is MEASURED and reported
    # rather than asserted away -- and it shrinks with dt (58 mK at 3.9 ms,
    # 10 mK at 2 ms on the reference case, against a rise of 45 K).
    def against_trend(values, rising: bool) -> float:
        pairs = zip(values, values[1:])
        return max(((a - b) if rising else (b - a) for a, b in pairs), default=0.0)

    powered = peaks[:burst_steps + 1] if burst_steps else peaks
    cooling = peaks[burst_steps + 1:] if burst_steps < len(peaks) - 1 else []
    step_artefact_k = max(against_trend(powered, rising=True),
                          against_trend(cooling, rising=False), 0.0)

    rise = steady_peak_c - start_peak_c
    # NOT the package's time constant: this is how fast the HOTTEST POINT
    # covers 63 % of its rise, which is much quicker than the bulk because a
    # hotspot is thin silicon reacting to power right underneath it. Both
    # numbers are reported, each called what it is.
    t63_peak_s = None
    if rise > 1e-9:
        target = start_peak_c + 0.632 * rise
        for t, peak in zip(times, peaks):
            if peak >= target:
                t63_peak_s = t
                break

    return {
        "ok": True,
        "dt_s": dt,
        "t_s": times,
        "peak_c": peaks,
        "chiplets": [{"id": cid, "peak_c": per_chiplet[cid]} for cid, *_ in rows],
        "idle_peak_c": start_peak_c,
        "burst_peak_c": burst_peak_c,
        "steady_peak_c": steady_peak_c,
        "tau_package_s": tau_s,
        "t63_peak_s": t63_peak_s,
        "step_artefact_k": step_artefact_k,
        "limit_c": limit,
        "time_to_limit_s": time_to_limit,
        "reached_fraction": ((burst_peak_c - start_peak_c) / rise) if rise > 1e-9 else 1.0,
        "profile": profile,
        "solve_s": time.perf_counter() - t_start,
    }


def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise SearchCancelled()


def solve_workers() -> int:
    """Threads to spread a batch of solves over: every core, within reason."""
    return max(1, min(MAX_SEARCH_THREADS, os.cpu_count() or 1))


def unit_fields(model: ThermalModel, chiplets, positions, sizes,
                cancel: Optional[threading.Event] = None,
                on_chiplet: Optional[Callable[[int], None]] = None) -> List[np.ndarray]:
    """Temperature rise, in kelvin, of each chiplet placed ALONE at each of its
    grid positions: array (positions, cells) per chiplet. One batched solve per
    chiplet against the cached factorization. Must be called under model.lock.
    """
    pkg = model.package
    n_cells = pkg.nx * pkg.ny * pkg.nz
    z0, z1 = model.power_z
    fields: List[np.ndarray] = []
    block = np.zeros((pkg.nx, pkg.ny, pkg.nz))
    # The rise is linear in power, so chiplets with the same footprint share
    # one set of solves at 1 W and are scaled: exact, and it removes repeated
    # work for identical blocks (four RISC-V cores cost one set, not four).
    per_watt: Dict[tuple, np.ndarray] = {}
    for index, (chiplet, (wc, hc), pos) in enumerate(zip(chiplets, sizes, positions)):
        _check_cancel(cancel)
        grid = chiplet.power_grid
        # Chiplets share solves only when their internal map matches too: a
        # MAC array in the middle is a different heat source from a uniform
        # block of the same size, and an imported map is its own thing. The
        # blocks themselves key this, not the cells they round to -- they are
        # no longer rounded.
        key = (wc, hc, chiplet.blocks,
               None if grid is None else (grid.cols, grid.rows, grid.weights))
        if key in per_watt:
            fields.append(per_watt[key] * chiplet.power_w)
            if on_chiplet is not None:
                on_chiplet(index + 1)
            continue
        nk = z1 - z0
        # One weights array covers both: an imported map, or the blocks and
        # whatever uniform power they leave. Blocks take the cells they
        # overlap rather than the cells they round to, so the same physical
        # hotspot is modelled at every mesh (see block_weights).
        weights = (grid_weights(wc, hc, grid) if grid is not None
                   else block_weights(wc, hc, chiplet.blocks))
        footprint = weights[:, :, None] / (nk * pkg.cell_volume_m3)
        rhs = np.zeros((n_cells, len(pos)), dtype=np.float64, order="F")
        for col, (x, y) in enumerate(pos):
            block[:] = 0.0
            block[x:x + wc, y:y + hc, z0:z1] = footprint
            rhs[:, col] = block.reshape(n_cells)
        # S theta = q: the rise above ambient; the boundary forcing cancels.
        full = model.factor.rise_for_power(rhs, workers=solve_workers())
        if full.size and float(full.min()) < -NEGATIVE_FIELD_TOL * max(1.0, float(full.max())):
            raise StudioError("a single-chiplet rise went negative; the exactness bound "
                              "would not hold, so the search refuses to run")
        theta_per_watt = np.ascontiguousarray(full[model.die_cells, :].T)
        per_watt[key] = theta_per_watt
        theta = theta_per_watt * chiplet.power_w
        peak = float(theta.max()) if theta.size else 0.0
        if theta.size and float(theta.min()) < -NEGATIVE_FIELD_TOL * max(1.0, peak):
            raise StudioError("a single-chiplet rise went negative; the exactness bound "
                              "would not hold, so the search refuses to run")
        fields.append(theta)
        if on_chiplet is not None:
            on_chiplet(index + 1)
    return fields


@dataclass
class SearchStats:
    nodes: int = 0
    leaves: int = 0
    pruned_overlap: int = 0
    pruned_symmetry: int = 0
    pruned_bound: int = 0


def branch_and_bound(fields: List[np.ndarray], positions: List[np.ndarray],
                     sizes: List[Tuple[int, int]], signatures: List[tuple],
                     incumbent: Tuple[float, Optional[List[int]]],
                     bound_cells: Optional[int] = None,
                     clearance_cells: int = 0,
                     improve_incumbent: bool = True,
                     cancel: Optional[threading.Event] = None,
                     on_progress: Optional[Callable[[SearchStats, float], None]] = None
                     ) -> Tuple[float, Optional[List[int]], SearchStats]:
    """Exact minimum, over every non-overlapping grid layout, of the peak rise.

    `fields[c]` is (positions, cells): chiplet c's rise at each grid position,
    over the cells where a peak can occur. Returns (best rise in K, position
    index per chiplet, stats). `incumbent` is a known layout (value, indices)
    or (inf, None); the search can only improve on it.

    `clearance_cells`: minimum empty cells between any two chiplets -- the
    assembly keep-out. Positions that violate it are excluded from the search
    exactly as overlapping ones are, so the optimum returned is the coolest
    BUILDABLE layout rather than the coolest drawable one.

    `bound_cells`: interior nodes only need a LOWER bound, and a maximum over
    any subset of cells is still one, so they use the first `bound_cells`
    columns (callers put the layers where peaks sit first). Complete layouts
    are always scored on every cell. Changing the subset changes the speed,
    never the answer.
    """
    k = len(fields)
    n_cells = fields[0].shape[1]
    nb = n_cells if bound_cells is None else max(1, min(int(bound_cells), n_cells))
    full = fields
    part = [np.ascontiguousarray(f[:, :nb]) for f in fields]
    self_peak = [float(f.max()) if f.size else 0.0 for f in fields]

    # Identical chiplets are interchangeable: keep them adjacent in the order
    # and force increasing position indices, which removes duplicate layouts
    # without losing any distinct one.
    groups: Dict[tuple, List[int]] = {}
    for i in range(k):
        groups.setdefault(signatures[i], []).append(i)
    order = [i for group in sorted(groups.values(), key=lambda g: -max(self_peak[i] for i in g))
             for i in group]

    compat: Dict[Tuple[int, int], np.ndarray] = {}
    for a in range(k):
        for b in range(k):
            if a == b:
                continue
            pa, pb = positions[a], positions[b]
            (wa, ha), (wb, hb) = sizes[a], sizes[b]
            ax, ay = pa[:, 0][:, None], pa[:, 1][:, None]
            bx, by = pb[:, 0][None, :], pb[:, 1][None, :]
            # Too close counts as incompatible, not just overlapping: the
            # pair must clear the keep-out in x or in y.
            overlap = ((np.minimum(ax + wa, bx + wb) + clearance_cells > np.maximum(ax, bx))
                       & (np.minimum(ay + ha, by + hb) + clearance_cells > np.maximum(ay, by)))
            compat[(a, b)] = ~overlap

    def allowed_for(ci: int, chosen_: List[Optional[int]]) -> np.ndarray:
        mask = np.ones(len(positions[ci]), dtype=bool)
        for cj in range(k):
            if cj != ci and chosen_[cj] is not None:
                mask &= compat[(cj, ci)][chosen_[cj]]
        return mask

    # A strong incumbent makes the bound bite early: greedy placement, then
    # move one chiplet at a time to its best position while that helps. Every
    # value is exact, so it is a real layout and a valid starting bound.
    greedy: List[Optional[int]] = [None] * k
    running_g = np.zeros(n_cells)
    feasible = improve_incumbent   # off only in tests that must prove the bare search
    for ci in (order if improve_incumbent else ()):
        idx = np.flatnonzero(allowed_for(ci, greedy))
        if idx.size == 0:
            feasible = False
            break
        j = int(idx[int(np.argmin((full[ci][idx] + running_g).max(axis=1)))])
        greedy[ci] = j
        running_g += full[ci][j]
    if feasible:
        value_g = float(running_g.max())
        for _ in range(50):
            improved = False
            for ci in range(k):
                others = running_g - full[ci][greedy[ci]]
                idx = np.flatnonzero(allowed_for(ci, greedy))
                peaks = (full[ci][idx] + others).max(axis=1)
                j = int(np.argmin(peaks))
                if peaks[j] < value_g - 1e-12:
                    greedy[ci] = int(idx[j])
                    running_g = others + full[ci][greedy[ci]]
                    value_g = float(peaks[j])
                    improved = True
            if not improved:
                break
        if value_g < incumbent[0]:
            incumbent = (value_g, list(greedy))

    minima = [f.min(axis=0) for f in part]
    rest = [np.zeros(nb) for _ in range(k + 1)]
    for d in range(k - 1, -1, -1):
        rest[d] = rest[d + 1] + minima[order[d]]

    widest = max(len(p) for p in positions)
    last = [time.perf_counter()]

    # The incumbent is shared: any thread may improve it, and every thread
    # reads it to prune. Ties are broken by the CHOICE, not by who got there
    # first, so the answer does not depend on how the work was divided.
    best_lock = threading.Lock()
    best = {"value": incumbent[0], "choice": incumbent[1]}

    def offer(value: float, choice: List[int]) -> None:
        with best_lock:
            if value < best["value"] - 1e-12 or (
                    abs(value - best["value"]) <= 1e-12
                    and (best["choice"] is None or choice < best["choice"])):
                best["value"], best["choice"] = value, list(choice)

    class Worker:
        """One thread's search state: its own buffers and its own counters,
        merged into the shared stats when it finishes."""

        def __init__(self) -> None:
            self.stats = SearchStats()
            self.chosen: List[Optional[int]] = [None] * k
            self.sums = [np.empty((widest, nb)) for _ in range(k)]
            self.scratch = [np.empty((widest, nb)) for _ in range(k)]

        def descend(self, d: int, run_full: np.ndarray, run_part: np.ndarray) -> None:
            chosen, stats = self.chosen, self.stats
            ci = order[d]
            count = len(positions[ci])
            allowed = np.ones(count, dtype=bool)
            for e in range(d):
                cj = order[e]
                allowed &= compat[(cj, ci)][chosen[cj]]
            stats.pruned_overlap += count - int(allowed.sum())
            if d > 0 and signatures[order[d - 1]] == signatures[ci]:
                before = int(allowed.sum())
                allowed[:chosen[order[d - 1]] + 1] = False
                stats.pruned_symmetry += before - int(allowed.sum())
            idx = np.flatnonzero(allowed)
            if idx.size == 0:
                return
            stats.nodes += 1
            now = time.perf_counter()
            if now - last[0] > 0.2:
                last[0] = now
                _check_cancel(cancel)
                if on_progress is not None:
                    on_progress(stats, best["value"])

            m = int(idx.size)
            partial = self.sums[d][:m]
            np.take(part[ci], idx, axis=0, out=partial)
            partial += run_part

            if d == k - 1:
                stats.leaves += m
                # The subset maximum is a lower bound on each layout's peak, so
                # only layouts whose subset maximum beats the best can beat it
                # in full.
                ceiling = best["value"]
                contenders = idx[partial.max(axis=1) < ceiling]
                if contenders.size:
                    peaks = (full[ci][contenders] + run_full).max(axis=1)
                    j = int(np.argmin(peaks))
                    if peaks[j] < ceiling:
                        chosen[ci] = int(contenders[j])
                        offer(float(peaks[j]), list(chosen))
                chosen[ci] = None
                return

            bound_rows = self.scratch[d][:m]
            np.add(partial, rest[d + 1], out=bound_rows)
            bound = bound_rows.max(axis=1)
            ranked = np.argsort(bound, kind="stable")
            for rank, j in enumerate(ranked):
                if bound[j] >= best["value"]:
                    stats.pruned_bound += int(ranked.size - rank)
                    break
                chosen[ci] = int(idx[j])
                self.descend(d + 1, run_full + full[ci][idx[j]], partial[j])
            chosen[ci] = None

    stats = SearchStats()

    def merge(worker: "Worker") -> None:
        stats.nodes += worker.stats.nodes
        stats.leaves += worker.stats.leaves
        stats.pruned_overlap += worker.stats.pruned_overlap
        stats.pruned_symmetry += worker.stats.pruned_symmetry
        stats.pruned_bound += worker.stats.pruned_bound

    # The root's children are independent subtrees, so they are what gets
    # spread across cores. Everything they share is read-only (the fields, the
    # compatibility masks, the bounds) except the incumbent, which only ever
    # improves -- a thread reading a stale one prunes less, never wrong.
    root = order[0]
    root_allowed = np.flatnonzero(np.ones(len(positions[root]), dtype=bool))
    root_bound = (part[root][root_allowed] + rest[1]).max(axis=1)
    root_order = [int(root_allowed[j]) for j in np.argsort(root_bound, kind="stable")]
    bound_by_index = {int(root_allowed[n]): float(root_bound[n]) for n in range(len(root_allowed))}

    workers = min(MAX_SEARCH_THREADS, max(1, os.cpu_count() or 1), len(root_order))
    if k < 2:
        workers = 1

    def run_root(child: int) -> "Worker":
        worker = Worker()
        if bound_by_index[child] >= best["value"]:
            worker.stats.pruned_bound += 1
            return worker
        worker.stats.nodes += 1
        worker.chosen[root] = child
        worker.descend(1, full[root][child].copy(), part[root][child].copy())
        worker.chosen[root] = None
        return worker

    if workers <= 1:
        single = Worker()
        single.descend(0, np.zeros(n_cells), np.zeros(nb))
        merge(single)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="risc3d-search") as pool:
            for worker in pool.map(run_root, root_order):
                merge(worker)

    if on_progress is not None:
        on_progress(stats, best["value"])
    return best["value"], best["choice"], stats


def optimize(cache: ModelCache, spec: PackageSpec, chiplets,
             on_progress: Optional[Callable[[dict], None]] = None,
             cancel: Optional[threading.Event] = None) -> dict:
    """Exact placement optimization, verified. See the module docstring."""
    t_start = time.perf_counter()

    def report(stage: str, **extra) -> None:
        if on_progress is not None:
            on_progress({"stage": stage, "elapsed_s": time.perf_counter() - t_start, **extra})

    report("preparing the model")
    model, _built = cache.get(spec)
    pkg = model.package
    catalog = _catalog(chiplets)
    keepout_m = spec.keepout_enforced_um * 1e-6
    violations = validate_placement(pkg, catalog, _placement(chiplets, spec.cell_mm), keepout_m)
    if violations:
        raise StudioError("the starting placement is invalid: " + "; ".join(violations))

    cell = spec.cell_mm
    pitch = _whole(spec.pitch_mm, cell)
    sizes = [(_whole(c.w_mm, cell), _whole(c.h_mm, cell)) for c in chiplets]
    positions: List[np.ndarray] = []
    for chiplet, (wc, hc) in zip(chiplets, sizes):
        pos = np.array([(x, y) for x in range(0, pkg.nx - wc + 1, pitch)
                        for y in range(0, pkg.ny - hc + 1, pitch)], dtype=np.int64)
        if pos.size == 0:
            raise StudioError(f"{chiplet.name} does not fit on the {spec.pitch_mm:g} mm grid")
        positions.append(pos)
    combinations = 1
    for pos in positions:
        combinations *= len(pos)

    # -- the starting placement, solved for real ----------------------------
    with model.lock:
        initial = solve_placement(model.solver, pkg, catalog,
                                  _placement(chiplets, cell), "initial")
        t0 = time.perf_counter()
        report("solving every chiplet at every grid position", done=0, total=len(chiplets),
               solves=int(sum(len(p) for p in positions)))
        fields = unit_fields(model, chiplets, positions, sizes, cancel,
                             on_chiplet=lambda n: report(
                                 "solving every chiplet at every grid position", done=n,
                                 total=len(chiplets),
                                 solves=int(sum(len(p) for p in positions))))
        precompute_s = time.perf_counter() - t0

    # -- incumbents: the starting layout (if on the grid) and a greedy one --
    incumbent_value, incumbent_choice = float("inf"), None
    start_choice = []
    for chiplet, pos in zip(chiplets, positions):
        xc, yc = _whole(chiplet.x_mm, cell), _whole(chiplet.y_mm, cell)
        hit = np.flatnonzero((pos[:, 0] == xc) & (pos[:, 1] == yc))
        start_choice.append(int(hit[0]) if hit.size else None)
    if all(c is not None for c in start_choice):
        value = float(sum(f[c] for f, c in zip(fields, start_choice)).max())
        incumbent_value, incumbent_choice = value, start_choice

    signatures = [(sizes[i], chiplets[i].power_w,
                   chiplets[i].blocks,
                   None if chiplets[i].power_grid is None
                   else (chiplets[i].power_grid.cols, chiplets[i].power_grid.rows,
                         chiplets[i].power_grid.weights))
                  for i in range(len(chiplets))]

    t0 = time.perf_counter()

    def search_progress(stats: SearchStats, best: float) -> None:
        report("searching every layout", nodes=stats.nodes, leaves=stats.leaves,
               best_c=(best + spec.ambient_c) if math.isfinite(best) else None,
               combinations=combinations)

    best_value, best_choice, stats = branch_and_bound(
        fields, positions, sizes, signatures, (incumbent_value, incumbent_choice),
        bound_cells=model.bound_cells, clearance_cells=spec.keepout_cells,
        cancel=cancel, on_progress=search_progress)
    search_s = time.perf_counter() - t0
    if best_choice is None:
        raise StudioError("no non-overlapping layout exists on this placement grid; "
                          "use a finer grid or a larger package")

    # -- the winner, re-solved from scratch ---------------------------------
    report("re-solving the winner with the full 3-D solver")
    winners = tuple(replace(c, x_mm=float(positions[i][best_choice[i]][0] * cell),
                            y_mm=float(positions[i][best_choice[i]][1] * cell))
                    for i, c in enumerate(chiplets))
    # Defense in depth: never report a layout that breaks the placement rules,
    # whatever the search believed.
    winner_violations = validate_placement(pkg, catalog, _placement(winners, cell), keepout_m)
    if winner_violations:
        raise StudioError("the search produced an invalid layout; refusing to report it: "
                          + "; ".join(winner_violations))
    with model.lock:
        optimized = solve_placement(model.solver, pkg, catalog,
                                    _placement(winners, cell), "optimized")
    verify_delta_k = abs(float(optimized.peak_c) - (best_value + spec.ambient_c))
    if verify_delta_k > RESOLVE_TOL_K:
        raise StudioError(f"the re-solved optimum differs from the search by "
                          f"{verify_delta_k:.2e} K; refusing to report it")

    kept_initial = float(initial.peak_c) <= float(optimized.peak_c) + 1e-9
    final_chiplets, final_result = ((chiplets, initial) if kept_initial
                                    else (winners, optimized))

    moves = []
    for before, after in zip(chiplets, final_chiplets):
        row_before = next(r for r in initial.per_chiplet if r.id == before.id)
        row_after = next(r for r in final_result.per_chiplet if r.id == after.id)
        moves.append({
            "id": before.id, "name": before.name,
            "from_mm": [before.x_mm, before.y_mm], "to_mm": [after.x_mm, after.y_mm],
            "distance_mm": math.hypot(after.x_mm - before.x_mm, after.y_mm - before.y_mm),
            "max_before_c": float(row_before.max_c), "max_after_c": float(row_after.max_c),
        })

    rise0 = float(initial.peak_c) - spec.ambient_c
    delta = float(initial.peak_c) - float(final_result.peak_c)
    symmetric = sum(len(g) - 1 for g in _groups(signatures).values())
    return {
        "ok": True,
        "initial": {**result_payload(model, chiplets, initial),
                    "placement": [c.as_dict() for c in chiplets]},
        "optimized": {**result_payload(model, final_chiplets, final_result),
                      "placement": [c.as_dict() for c in final_chiplets]},
        "moves": moves,
        "kept_initial": kept_initial,
        "improvement": {"delta_k": delta, "pct_of_rise": 100.0 * delta / rise0 if rise0 > 0 else 0.0,
                        "initial_peak_c": float(initial.peak_c),
                        "optimized_peak_c": float(final_result.peak_c)},
        "certificate": {
            "exact": True,
            "pitch_mm": spec.pitch_mm,
            "keepout_um": spec.keepout_um,
            "keepout_enforced_um": spec.keepout_enforced_um,
            "positions_per_chiplet": [int(len(p)) for p in positions],
            "combinations": int(combinations),
            "layouts_scored": int(stats.leaves),
            "search_nodes": int(stats.nodes),
            "excluded_overlap": int(stats.pruned_overlap),
            "excluded_symmetry": int(stats.pruned_symmetry),
            "excluded_bound": int(stats.pruned_bound),
            "identical_chiplets": int(symmetric),
            "unit_solves": int(sum(len(p) for p in positions)),
            "precompute_s": precompute_s,
            "search_s": search_s,
            "total_s": time.perf_counter() - t_start,
            "lattice_optimum_c": best_value + spec.ambient_c,
            "resolve_delta_k": verify_delta_k,
        },
    }


def _groups(signatures) -> Dict[tuple, List[int]]:
    groups: Dict[tuple, List[int]] = {}
    for i, sig in enumerate(signatures):
        groups.setdefault(sig, []).append(i)
    return groups
