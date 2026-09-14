"""
experiment.py
RISC3D-AI :: one placement-optimization experiment, end to end.

    catalog (WHAT each chiplet is) + initial placement (WHERE it starts)
        -> placement validation
        -> spatial power map: each chiplet's watts over its own footprint
        -> real thermal solve                 [the immutable baseline]
        -> optimizer over the placement lattice
        -> optimized placement
        -> real thermal RE-solve              [authoritative, never an estimate]
        -> before / after comparison

This is the PRODUCT layer. It sits on top of the protected numerical core
(thermal_solver, optimizer, exhaustive, composite_materials), calls it
through its public entry points, and reimplements none of its physics.

Units are SI throughout (m, W, K). Micrometres and millimetres appear only
where a person writes or reads a number: the config file and report strings.

WHY THIS MODULE EXISTS
The pipeline could already answer "how hot does this get?" but not "what was
the starting placement, and what did the optimizer change?". Stage 1 put one
uniform power footprint on the die; Stage 2 placed a chiplet catalog from a
starting layout nobody specified -- the optimizer's own shelf packing --
whose origins did not even lie on the lattice the search used. One run
therefore reported two different "initial" temperatures. Here the initial
placement is an input, the baseline is solved from it, and the optimized
placement is re-solved by the real solver before it is reported.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config_loader import ConfigError, PipelineConfig
from src.grid_builder import GridBuilder, GridBuilderError
from src.thermal_solver import ThermalSolver, ThermalSolverError
from src.optimizer import ChipletPlacementOptimizer, ChipletSpec, OptimizerError
from src.exhaustive import ExhaustivePlacementSearch, ExhaustiveSearchError

# Materials of the three-region stack, bottom to top. Shared with main.py so
# the legacy Stage 1 stack and this one cannot drift apart.
STACK_MATERIALS = ("silicon", "bump_underfill_composite", "silicon")

# A length counts as "a whole number of cells (or lattice steps)" within this
# relative tolerance. Config lengths are micrometre floats and the spacings
# are metre floats, so an exact ratio still lands a few ulps off an integer.
GEOM_REL_TOL = 1e-9

# Deposited power must match specified power to this relative tolerance. The
# only error is the floating-point sum over ~10^4 cells.
POWER_REL_TOL = 1e-9

# The re-solved optimum must agree with the peak the optimizer reported for
# the same layout. Both are direct solves of the same operator with the same
# q_map, so they agree to round-off; this is orders looser.
RESOLVE_TOL_K = 1e-6

OBJECTIVE = ("minimize the steady-state peak temperature: the maximum over every "
             "cell of the solved field")

CONSTRAINTS = (
    "every chiplet lies inside the package",
    "no two chiplets overlap",
    "every origin lies on the placement lattice",
    "chiplet footprints and powers are fixed inputs, never search variables",
    "package, stack, materials and boundary conditions are fixed inputs",
)


class ExperimentError(Exception):
    """Raised when an experiment cannot be built, is not physically valid, or
    fails one of its own consistency checks."""


# ---------------------------------------------------------------------------
# Data model: WHAT the chiplets are, WHERE they are, and what holds them
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PowerBlock:
    """A region INSIDE a chiplet that carries more than its share of the power.

    A chiplet is not a uniform heater: an NPU's MAC array, a GPU's shader
    cluster or a CPU's core complex occupies part of the die and draws most
    of the current, and that is where the junction actually runs hottest.
    Blocks are stored as fractions of the footprint so they survive a resize
    or a rotation, and `share` is the fraction of the chiplet's power the
    block carries. Whatever the blocks do not take is spread over the rest of
    the footprint, so the chiplet's total power is unchanged.
    """
    x_frac: float
    y_frac: float
    w_frac: float
    h_frac: float
    share: float

    def rotated(self) -> "PowerBlock":
        """The block as it sits after the chiplet is rotated 90 degrees.

        A point (u, v) of the old footprint lands at (1 - v, u) of the new
        one, so the block's corners swap axes and its x runs backwards.
        """
        return PowerBlock(1.0 - self.y_frac - self.h_frac, self.x_frac,
                          self.h_frac, self.w_frac, self.share)


@dataclass(frozen=True)
class PowerGrid:
    """A measured power map across a chiplet: a grid of relative weights.

    This is what a design team actually has -- a per-block or per-tile power
    table exported from their own tools -- rather than the tidy rectangles of
    PowerBlock. The weights are RELATIVE: they are normalized over the mesh
    cells the chiplet covers, so the chiplet's total power is whatever the
    catalog says it is, however the file was scaled (W, mW, percentages, or
    activity counts).

    Row order is the one a spreadsheet shows: `weights[0]` is the TOP row of
    the chiplet, the high-y edge. The app draws the map it imported so a file
    that is upside down is visible immediately rather than believed.
    """
    cols: int
    rows: int
    weights: Tuple[float, ...]          # row-major, len == cols * rows

    def rotated(self) -> "PowerGrid":
        """The map as it sits after the chiplet is rotated 90 degrees."""
        grid = [self.weights[r * self.cols:(r + 1) * self.cols] for r in range(self.rows)]
        turned = [tuple(row[c] for row in grid) for c in range(self.cols - 1, -1, -1)]
        return PowerGrid(cols=self.rows, rows=self.cols,
                         weights=tuple(v for row in turned for v in row))


@dataclass(frozen=True)
class ChipletDef:
    """WHAT a chiplet is: a rectangular footprint carrying a fixed power.

    Authored in micrometres, like the config and optimizer.ChipletSpec; every
    calculation uses the SI properties below. Carries no position: the same
    chiplet can be placed many ways, and that separation is what makes an
    optimization expressible as a move from one Placement to another.
    """
    id: str
    name: str
    width_um: float
    height_um: float
    power_w: float
    blocks: Tuple[PowerBlock, ...] = ()      # internal hotspots; empty means uniform
    power_grid: Optional["PowerGrid"] = None  # an imported map; wins over blocks

    @property
    def width_m(self) -> float:
        return self.width_um * 1e-6

    @property
    def height_m(self) -> float:
        return self.height_um * 1e-6

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def area_mm2(self) -> float:
        return self.width_um * self.height_um * 1e-6

    @property
    def power_density_w_cm2(self) -> float:
        area_cm2 = self.width_um * self.height_um * 1e-8
        return self.power_w / area_cm2 if area_cm2 > 0 else float("nan")

    def to_spec(self) -> ChipletSpec:
        """The optimizer's catalog type. Named by id, so a returned layout
        maps back to exactly one chiplet, and given the micrometre values
        verbatim, so the optimizer meshes the same rectangle this module does.
        """
        return ChipletSpec(name=self.id, width_um=self.width_um,
                           depth_um=self.height_um, power_w=self.power_w)


@dataclass(frozen=True)
class Placement:
    """WHERE each chiplet sits: lower-left origins in metres, catalog order."""
    pitch_m: float
    origins_m: Tuple[Tuple[str, float, float], ...]

    def origin(self, chiplet_id: str) -> Tuple[float, float]:
        for cid, x_m, y_m in self.origins_m:
            if cid == chiplet_id:
                return (x_m, y_m)
        raise ExperimentError(f"placement has no origin for chiplet '{chiplet_id}'")

    @property
    def ids(self) -> Tuple[str, ...]:
        return tuple(cid for cid, _x, _y in self.origins_m)


@dataclass(frozen=True)
class Package:
    """The physical object the chiplets sit in, and the mesh that resolves it."""
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    die_z: Tuple[int, int]                  # [start, end) cells of the die layer
    layer_cells: Tuple[int, int, int]       # interposer / composite / die
    stack_um: Tuple[float, float, float]
    sink_face: str
    sink_bc: str
    h_convective: float
    t_ambient_c: float

    @property
    def lx_m(self) -> float:
        return self.nx * self.dx

    @property
    def ly_m(self) -> float:
        return self.ny * self.dy

    @property
    def lz_m(self) -> float:
        return self.nz * self.dz

    @property
    def cell_volume_m3(self) -> float:
        return self.dx * self.dy * self.dz

    @property
    def sink_area_m2(self) -> float:
        return self.lx_m * self.ly_m

    @property
    def t_ambient_k(self) -> float:
        return self.t_ambient_c + 273.15


def package_from_config(config: PipelineConfig) -> Package:
    g, bcs, stack = config.grid, config.boundary_conditions, config.stack
    # Raises unless the mesh holds the authored stack exactly, in whole cells.
    n_i, n_b, n_d = stack.layer_cells(g.nz, g.dz)
    sink_face = "bottom" if bcs.bottom_bc != "neumann" else "top"
    return Package(
        nx=g.nx, ny=g.ny, nz=g.nz, dx=g.dx, dy=g.dy, dz=g.dz,
        die_z=(n_i + n_b, g.nz), layer_cells=(n_i, n_b, n_d),
        stack_um=(stack.interposer_um, stack.bump_um, stack.die_um),
        sink_face=sink_face,
        sink_bc=(bcs.bottom_bc if sink_face == "bottom" else bcs.top_bc),
        h_convective=bcs.h_convective, t_ambient_c=bcs.t_ambient_c,
    )


def catalog_from_config(config: PipelineConfig) -> Tuple[ChipletDef, ...]:
    if not config.chiplets:
        raise ExperimentError(
            "this configuration defines no chiplets, so there is no placement "
            "experiment to run; add 'chiplets' and 'initial_placement' to the config")
    return tuple(ChipletDef(id=c.id, name=c.name, width_um=c.width_um,
                            height_um=c.height_um, power_w=c.power_w)
                 for c in config.chiplets)


def initial_placement_from_config(config: PipelineConfig) -> Placement:
    placement = config.initial_placement
    if placement is None:
        raise ExperimentError("this configuration defines no initial_placement")
    origins = tuple((c.id,
                     placement.origins_um[c.id][0] * 1e-6,
                     placement.origins_um[c.id][1] * 1e-6)
                    for c in config.chiplets)
    return Placement(pitch_m=placement.pitch_um * 1e-6, origins_m=origins)


# ---------------------------------------------------------------------------
# Placement validation
# ---------------------------------------------------------------------------

def _is_whole(value_m: float, unit_m: float) -> bool:
    if unit_m <= 0:
        return False
    n = value_m / unit_m
    return abs(n - round(n)) <= GEOM_REL_TOL * max(1.0, abs(n))


def footprint_cells(package: Package, chiplet: ChipletDef) -> Tuple[int, int]:
    """Voxel extent of a footprint, from the optimizer's own resolver, so this
    module and the optimizer always mesh the identical rectangle."""
    return chiplet.to_spec().extent_vox(package.dx, package.dy)


def origin_cells(package: Package, x_m: float, y_m: float) -> Tuple[int, int]:
    return int(round(x_m / package.dx)), int(round(y_m / package.dy))


def validate_placement(package: Package, catalog: Tuple[ChipletDef, ...],
                       placement: Placement, keepout_m: float = 0.0) -> List[str]:
    """Every physical constraint a placement must satisfy, as a list of
    violations in plain language. An empty list means valid.

    `keepout_m` is the minimum edge-to-edge clearance between any two
    chiplets -- assembly needs room for underfill to flow and for microbump
    dispense tolerance, so touching neighbours are not buildable even though
    they are geometrically disjoint. The rule is the standard spacing one:
    two chiplets are legal when they clear each other by at least the
    keep-out in x OR in y, which also refuses corner-to-corner contact.

    Returned rather than raised: the caller decides whether an invalid
    placement is a refusal to run (the initial one) or a defect in the
    optimizer (an optimized one), and both want the whole list rather than
    the first failure.
    """
    violations: List[str] = []
    ids = [c.id for c in catalog]
    if sorted(ids) != sorted(placement.ids):
        violations.append(
            f"placement covers {sorted(placement.ids)} but the catalog is {sorted(ids)}")
        return violations

    tol_m = GEOM_REL_TOL * max(package.lx_m, package.ly_m)
    rects: Dict[str, Tuple[float, float, float, float]] = {}
    for c in catalog:
        x_m, y_m = placement.origin(c.id)
        nvx, nvy = footprint_cells(package, c)
        if not _is_whole(c.width_m, package.dx) or not _is_whole(c.height_m, package.dy):
            violations.append(
                f"{c.id}: footprint {c.width_um:.1f} x {c.height_um:.1f} um is not a whole "
                f"number of {package.dx * 1e6:.1f} x {package.dy * 1e6:.1f} um cells; it would "
                f"be meshed as {nvx} x {nvy} cells "
                f"({nvx * package.dx * 1e6:.1f} x {nvy * package.dy * 1e6:.1f} um)")
        if not _is_whole(x_m, package.dx) or not _is_whole(y_m, package.dy):
            violations.append(
                f"{c.id}: origin ({x_m * 1e6:.1f}, {y_m * 1e6:.1f}) um is not on a cell "
                f"boundary of the {package.dx * 1e6:.1f} x {package.dy * 1e6:.1f} um mesh")
        if not _is_whole(x_m, placement.pitch_m) or not _is_whole(y_m, placement.pitch_m):
            violations.append(
                f"{c.id}: origin ({x_m * 1e6:.1f}, {y_m * 1e6:.1f}) um is off the "
                f"{placement.pitch_m * 1e6:.0f} um placement lattice")
        if c.power_w < 0:
            violations.append(f"{c.id}: power {c.power_w} W is negative")
        x1_m, y1_m = x_m + c.width_m, y_m + c.height_m
        if (x_m < -tol_m or y_m < -tol_m
                or x1_m > package.lx_m + tol_m or y1_m > package.ly_m + tol_m):
            violations.append(
                f"{c.id}: footprint x[{x_m * 1e6:.0f}..{x1_m * 1e6:.0f}] "
                f"y[{y_m * 1e6:.0f}..{y1_m * 1e6:.0f}] um leaves the "
                f"{package.lx_m * 1e6:.0f} x {package.ly_m * 1e6:.0f} um package")
        rects[c.id] = (x_m, y_m, x1_m, y1_m)

    for a, b in itertools.combinations(ids, 2):
        ax0, ay0, ax1, ay1 = rects[a]
        bx0, by0, bx1, by1 = rects[b]
        ox = min(ax1, bx1) - max(ax0, bx0)
        oy = min(ay1, by1) - max(ay0, by0)
        if ox > tol_m and oy > tol_m:
            violations.append(
                f"{a} and {b} overlap by {ox * 1e6:.1f} x {oy * 1e6:.1f} um")
        elif keepout_m > 0.0 and ox > tol_m - keepout_m and oy > tol_m - keepout_m:
            # Disjoint, but not by enough: the clearance is the larger of the
            # two axis gaps, since clearing in one axis is what assembly needs.
            clearance = max(max(-ox, -oy), 0.0)
            violations.append(
                f"{a} and {b} clear each other by only {clearance * 1e6:.0f} um; "
                f"the keep-out is {keepout_m * 1e6:.0f} um")
    return violations


def neighbour_and_edge_gaps_m(package: Package, catalog: Tuple[ChipletDef, ...],
                              placement: Placement, chiplet_id: str) -> Tuple[float, float]:
    """(nearest-neighbour edge-to-edge gap, distance to the nearest package
    edge), in metres. Both are quantities a placement can change, which is
    why the before/after comparison reports them."""
    me = next(c for c in catalog if c.id == chiplet_id)
    x0, y0 = placement.origin(chiplet_id)
    x1, y1 = x0 + me.width_m, y0 + me.height_m
    edge = min(x0, y0, package.lx_m - x1, package.ly_m - y1)
    gap = float("inf")
    for other in catalog:
        if other.id == chiplet_id:
            continue
        ox0, oy0 = placement.origin(other.id)
        ox1, oy1 = ox0 + other.width_m, oy0 + other.height_m
        gap = min(gap, math.hypot(max(ox0 - x1, x0 - ox1, 0.0),
                                  max(oy0 - y1, y0 - oy1, 0.0)))
    return gap, edge


# ---------------------------------------------------------------------------
# Power map: chiplet geometry + power -> cell-level source Q
# ---------------------------------------------------------------------------

POWER_PLANES = ("active", "volume")
POWER_PLANE_DEFAULT = "active"


def power_z_range(package: "Package", plane: str = POWER_PLANE_DEFAULT) -> Tuple[int, int]:
    """The [start, end) die cells a chiplet's heat is deposited in.

    "active" -- the device layer: ONE cell at the bumped face of the die. A
    flip-chip die generates its heat in a device layer under a micrometre
    thick, sitting against the bumps, with the sink on the opposite
    (backside) face. The mesh cannot resolve a micrometre, so the source is
    one cell (dz) thick, but it sits at the correct face, which is what sets
    the conduction path to the sink.

    "volume" -- power spread evenly through the whole die thickness. This was
    the earlier convention; it is kept so the two can be compared, and is
    what the transient warm-up mass of the older results assumed.
    """
    if plane not in POWER_PLANES:
        raise ExperimentError(f"power plane must be one of {POWER_PLANES}, got {plane!r}")
    z0, z1 = package.die_z
    return (z0, min(z0 + 1, z1)) if plane == "active" else (z0, z1)


@dataclass(frozen=True)
class PowerMap:
    """The cell-level heat source, and the evidence that it is the catalog's.

    q_w_m3 is volumetric power density, which is what ThermalSolver consumes.
    Each chiplet's watts go only into the cells its own footprint covers, and
    only in the die layer; nothing is spread across unrelated die regions.
    """
    q_w_m3: np.ndarray
    per_chiplet_w: Tuple[Tuple[str, float], ...]
    specified_w: float
    deposited_w: float
    blocks: Tuple[Tuple[str, int, int, int, int], ...]   # id, i0, j0, nvx, nvy
    plane: str = POWER_PLANE_DEFAULT
    plane_z: Tuple[int, int] = (0, 0)

    @property
    def powered_cells(self) -> int:
        return int(np.count_nonzero(self.q_w_m3))


def grid_weights(nvx: int, nvy: int, grid: PowerGrid) -> np.ndarray:
    """The imported map resampled onto the mesh cells, normalized to sum 1.

    Each mesh cell takes the value of the map cell its CENTRE falls in --
    nearest-tile sampling, not interpolation, because a power map is a
    statement about regions rather than a smooth field, and averaging
    neighbouring tiles would cool the hotspot the file exists to describe.
    Normalizing over the mesh cells afterwards is what makes the chiplet's
    total power exact whatever the mesh does to the map.
    """
    if grid.cols <= 0 or grid.rows <= 0:
        raise ExperimentError("a power grid needs at least one column and one row")
    if len(grid.weights) != grid.cols * grid.rows:
        raise ExperimentError(
            f"power grid says {grid.cols} x {grid.rows} but carries {len(grid.weights)} values")
    values = np.asarray(grid.weights, dtype=np.float64).reshape(grid.rows, grid.cols)
    if np.any(values < 0.0):
        raise ExperimentError("a power grid cannot contain negative power")
    total = float(values.sum())
    if total <= 0.0:
        raise ExperimentError("a power grid must carry some power")

    i = np.arange(nvx)
    j = np.arange(nvy)
    col = np.clip(((i + 0.5) / nvx * grid.cols).astype(int), 0, grid.cols - 1)
    # Mesh j counts up from the chiplet's bottom edge; the map's first row is
    # its top edge, so the row index runs the other way.
    row = np.clip(((1.0 - (j + 0.5) / nvy) * grid.rows).astype(int), 0, grid.rows - 1)
    cells = values[np.ix_(row, col)].T          # (nvx, nvy)
    cells_sum = float(cells.sum())
    if cells_sum <= 0.0:
        raise ExperimentError(
            "every mesh cell landed on an empty tile of the power grid; the map is "
            "finer than the mesh can resolve")
    return cells / cells_sum


def block_cells(nvx: int, nvy: int, blocks: Tuple[PowerBlock, ...]
                ) -> Tuple[Tuple[int, int, int, int, float], ...]:
    """Internal blocks as whole cells: (i, j, cells_x, cells_y, share).

    A block is rounded to the cells it covers and never rounds away to
    nothing -- a MAC array smaller than one cell still has to go somewhere,
    and dropping it would quietly move its power back into the uniform part.
    The cells are relative to the chiplet's own origin.
    """
    out = []
    for b in blocks:
        i0 = min(nvx - 1, max(0, int(round(b.x_frac * nvx))))
        j0 = min(nvy - 1, max(0, int(round(b.y_frac * nvy))))
        w = max(1, min(nvx - i0, int(round(b.w_frac * nvx))))
        h = max(1, min(nvy - j0, int(round(b.h_frac * nvy))))
        out.append((i0, j0, w, h, b.share))
    return tuple(out)


def uniform_share(blocks: Tuple[PowerBlock, ...]) -> float:
    """The fraction of a chiplet's power that is NOT in a block."""
    return 1.0 - sum(b.share for b in blocks)


def build_power_map(package: Package, catalog: Tuple[ChipletDef, ...],
                    placement: Placement,
                    plane: str = POWER_PLANE_DEFAULT) -> PowerMap:
    """Builds Q from geometry: for each chiplet, its power divided by the
    volume of the cells its own footprint covers through the die layer.

    Total power is conserved by construction and then MEASURED back off the
    array (POWER_REL_TOL), so a footprint rounding error can never quietly
    create or destroy watts.
    """
    z0, z1 = power_z_range(package, plane)
    nk = z1 - z0
    if nk <= 0:
        raise ExperimentError(f"die layer is empty: z[{z0}:{z1}]")

    q = np.zeros((package.nx, package.ny, package.nz), dtype=np.float64)
    blocks: List[Tuple[str, int, int, int, int]] = []
    for c in catalog:
        x_m, y_m = placement.origin(c.id)
        i0, j0 = origin_cells(package, x_m, y_m)
        nvx, nvy = footprint_cells(package, c)
        if i0 < 0 or j0 < 0 or i0 + nvx > package.nx or j0 + nvy > package.ny:
            raise ExperimentError(
                f"'{c.id}' does not fit in the mesh at cell ({i0}, {j0}) with extent "
                f"{nvx} x {nvy}; call validate_placement() first")
        if c.power_grid is not None:
            # An imported map replaces the block description entirely.
            weights = grid_weights(nvx, nvy, c.power_grid)
            q[i0:i0 + nvx, j0:j0 + nvy, z0:z1] += (
                c.power_w * weights[:, :, None] / (nk * package.cell_volume_m3))
            blocks.append((c.id, i0, j0, nvx, nvy))
            continue
        # The part of the power that is spread over the whole footprint...
        rest = uniform_share(c.blocks)
        if rest < -1e-9:
            raise ExperimentError(
                f"{c.id}: internal blocks claim {1.0 - rest:.3f} of the power, which is "
                f"more than the chiplet has")
        n_cells = nvx * nvy * nk
        q[i0:i0 + nvx, j0:j0 + nvy, z0:z1] += c.power_w * rest / (n_cells * package.cell_volume_m3)
        # ... and each internal block's own share, over its own cells.
        for bi, bj, bw, bh, share in block_cells(nvx, nvy, c.blocks):
            cells = bw * bh * nk
            q[i0 + bi:i0 + bi + bw, j0 + bj:j0 + bj + bh, z0:z1] += (
                c.power_w * share / (cells * package.cell_volume_m3))
        blocks.append((c.id, i0, j0, nvx, nvy))

    per_chiplet = tuple(
        (cid, float(np.sum(q[i0:i0 + nvx, j0:j0 + nvy, z0:z1])) * package.cell_volume_m3)
        for cid, i0, j0, nvx, nvy in blocks)
    specified = float(sum(c.power_w for c in catalog))
    deposited = float(np.sum(q)) * package.cell_volume_m3
    if abs(deposited - specified) > POWER_REL_TOL * max(specified, 1e-30):
        raise ExperimentError(
            f"power map deposits {deposited!r} W but the catalog specifies {specified!r} W "
            f"(relative tolerance {POWER_REL_TOL:.0e})")

    q.setflags(write=False)
    return PowerMap(q_w_m3=q, per_chiplet_w=per_chiplet, specified_w=specified,
                    deposited_w=deposited, blocks=tuple(blocks), plane=plane,
                    plane_z=(z0, z1))


def build_stack(config: PipelineConfig) -> GridBuilder:
    """The three-region stack: materials and boundary tags, and NO power.

    Power is a separate, explicit step (build_power_map), which is what lets
    one stack serve many placements without rebuilding materials, and what
    keeps "where the heat is" out of "what the package is made of".
    """
    gb = GridBuilder(config=config.to_dict())
    n_i, n_b, _n_d = config.stack.layer_cells(config.grid.nz, config.grid.dz)
    gb.assign_layer(slice(0, n_i), STACK_MATERIALS[0])
    gb.assign_layer(slice(n_i, n_i + n_b), STACK_MATERIALS[1])
    gb.assign_layer(slice(n_i + n_b, config.grid.nz), STACK_MATERIALS[2])
    gb.tag_boundaries()
    return gb


# ---------------------------------------------------------------------------
# Solving one placement
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChipletThermal:
    id: str
    max_c: float
    mean_c: float


@dataclass(frozen=True)
class ThermalResult:
    """One solved placement, with the checks that were run on it."""
    label: str
    placement: Placement
    power: PowerMap
    field_k: np.ndarray
    peak_c: float
    peak_ijk: Tuple[int, int, int]
    peak_center_m: Tuple[float, float, float]
    ambient_c: float
    residual: float
    energy_imbalance_k: float
    sink_heat_w: float
    per_chiplet: Tuple[ChipletThermal, ...]
    solve_s: float

    @property
    def rise_k(self) -> float:
        return self.peak_c - self.ambient_c

    def chiplet(self, chiplet_id: str) -> ChipletThermal:
        for row in self.per_chiplet:
            if row.id == chiplet_id:
                return row
        raise ExperimentError(f"no thermal record for chiplet '{chiplet_id}'")

    @property
    def hottest_chiplet_id(self) -> str:
        return max(self.per_chiplet, key=lambda r: r.max_c).id


def sink_heat_w(package: Package, solver: ThermalSolver, field_k: np.ndarray) -> float:
    """Heat leaving through the sink face, computed HERE from the boundary
    condition and the solved field. It shares no code with the solver's own
    energy check, so agreement between the two is evidence, not tautology.
    """
    k = package.nz - 1 if package.sink_face == "top" else 0
    rise = field_k[:, :, k] - package.t_ambient_k
    area = package.dx * package.dy
    if package.sink_bc == "robin":
        # The film in series with conduction across the half cell between the
        # last cell centre and the cooled surface, matching the boundary the
        # solver assembles -- written out here from the physics rather than
        # read from the solver, so agreement remains evidence.
        resistance = package.dz / (2.0 * solver.kz_map[:, :, k]) + 1.0 / package.h_convective
        return float(np.sum(area * rise / resistance))
    # Dirichlet: conduction across the half cell between the last cell centre
    # and the face held at ambient.
    conductance = 2.0 * solver.kz_map[:, :, k] * area / package.dz
    return float(np.sum(conductance * rise))


def solve_placement(solver: ThermalSolver, package: Package,
                    catalog: Tuple[ChipletDef, ...], placement: Placement,
                    label: str, plane: str = POWER_PLANE_DEFAULT) -> ThermalResult:
    """Solves ONE placement with the real thermal solver.

    solve_steady_state() raises unless its residual, energy-balance and
    convective-floor checks pass, so a returned result is a verified one. The
    checks are re-evaluated afterwards purely to report their VALUES: the
    solver computes them, raises on failure, and logs but does not return
    them. That read-only use of two private members is deliberate and pinned
    by a test; nothing here mutates solver state beyond q_map.
    """
    power = build_power_map(package, catalog, placement, plane)
    solver.q_map[:] = power.q_w_m3            # the one input a placement changes

    t0 = time.perf_counter()
    try:
        field = solver.solve_steady_state(apply_result=False)
    except ThermalSolverError as exc:
        raise ExperimentError(
            f"steady-state solve failed for the {label} placement: {exc}") from exc
    solve_s = time.perf_counter() - t0

    n = field.size
    forcing = solver._boundary_forcing + solver.q_map.reshape(n) / solver._rho_cp.reshape(n)
    checks = solver._verify_steady_state(field.reshape(n), forcing)

    i, j, k = (int(v) for v in np.unravel_index(int(np.argmax(field)), field.shape))
    z0, z1 = package.die_z
    per_chiplet = tuple(
        ChipletThermal(
            id=cid,
            max_c=float(np.max(field[i0:i0 + nvx, j0:j0 + nvy, z0:z1])) - 273.15,
            mean_c=float(np.mean(field[i0:i0 + nvx, j0:j0 + nvy, z0:z1])) - 273.15,
        )
        for cid, i0, j0, nvx, nvy in power.blocks)

    stored = np.array(field, dtype=np.float64)
    stored.setflags(write=False)
    return ThermalResult(
        label=label, placement=placement, power=power, field_k=stored,
        peak_c=float(field[i, j, k]) - 273.15, peak_ijk=(i, j, k),
        peak_center_m=((i + 0.5) * package.dx, (j + 0.5) * package.dy,
                       (k + 0.5) * package.dz),
        ambient_c=package.t_ambient_c,
        residual=float(checks["residual"]),
        energy_imbalance_k=float(checks["energy_imbalance_k"]),
        sink_heat_w=sink_heat_w(package, solver, field),
        per_chiplet=per_chiplet, solve_s=solve_s,
    )


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OptimizationRecord:
    """What the optimizer was asked, what it did, and how its answer held up
    against the real solver."""
    method: str
    objective: str
    constraints: Tuple[str, ...]
    pitch_m: float
    optimizer_peak_c: float
    runtime_s: float
    candidates_evaluated: int
    candidates_total: Optional[int]
    rejected_or_pruned: int
    precompute_solves: Optional[int]
    superposition_error_k: Optional[float]
    seed: Optional[int]
    deterministic: bool
    history_c: Tuple[float, ...]
    resolve_delta_k: float
    resolve_field_delta_k: float


@dataclass(frozen=True)
class MoveRow:
    """One chiplet's before and after, in the quantities a placement changes."""
    id: str
    name: str
    x0_m: float
    y0_m: float
    x1_m: float
    y1_m: float
    peak0_c: float
    peak1_c: float
    gap0_m: float
    gap1_m: float
    edge0_m: float
    edge1_m: float

    @property
    def dx_m(self) -> float:
        return self.x1_m - self.x0_m

    @property
    def dy_m(self) -> float:
        return self.y1_m - self.y0_m

    @property
    def distance_m(self) -> float:
        return math.hypot(self.dx_m, self.dy_m)

    @property
    def moved(self) -> bool:
        return self.distance_m > 0.0


class Experiment:
    """One experiment: a catalog, an initial placement, and everything that
    follows. The initial result is computed once and never overwritten --
    optimization writes to separate fields, and every stored array is
    read-only, so a "before" cannot silently become an "after".
    """

    def __init__(self, config: PipelineConfig, source: str = ""):
        self.config = config
        self.source = source
        self.package = package_from_config(config)
        self.catalog = catalog_from_config(config)
        self.initial_placement = initial_placement_from_config(config)
        self.violations = validate_placement(self.package, self.catalog,
                                             self.initial_placement)
        self.grid_builder: Optional[GridBuilder] = None
        self.solver: Optional[ThermalSolver] = None
        self.initial: Optional[ThermalResult] = None
        self.optimized: Optional[ThermalResult] = None
        self.optimization: Optional[OptimizationRecord] = None
        # Set when the screening surrogate ran, so the report can show what it
        # ranked with and how well it ranked.
        self.surrogate = None
        self.build_s = 0.0
        self.solver_init_s = 0.0

    # -- initial ---------------------------------------------------------
    def solve_initial(self) -> ThermalResult:
        if self.violations:
            raise ExperimentError("initial placement is invalid: "
                                  + "; ".join(self.violations))
        t0 = time.perf_counter()
        try:
            self.grid_builder = build_stack(self.config)
            self.grid_builder.validate_grid()
        except (GridBuilderError, ConfigError) as exc:
            raise ExperimentError(f"stack assembly failed: {exc}") from exc
        self.build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        try:
            self.solver = ThermalSolver.from_grid_builder(self.grid_builder)
        except ThermalSolverError as exc:
            raise ExperimentError(f"solver initialization failed: {exc}") from exc
        self.solver_init_s = time.perf_counter() - t0

        self.initial = solve_placement(self.solver, self.package, self.catalog,
                                       self.initial_placement, "initial")
        return self.initial

    # -- optimization ----------------------------------------------------
    def layout_of(self, placement: Placement) -> Tuple[Tuple[int, int], ...]:
        return tuple(origin_cells(self.package, *placement.origin(c.id))
                     for c in self.catalog)

    def placement_of(self, layout) -> Placement:
        return Placement(
            pitch_m=self.initial_placement.pitch_m,
            origins_m=tuple((c.id, x0 * self.package.dx, y0 * self.package.dy)
                            for c, (x0, y0) in zip(self.catalog, layout)))

    def optimize(self, method: str = "exhaustive", iterations: int = 30,
                 seed: int = 0, verbose: bool = True, surrogate_top_k: int = 25,
                 surrogate_samples: int = 40) -> ThermalResult:
        if self.initial is None:
            raise ExperimentError(
                "solve_initial() must run first: the optimizer starts from the initial "
                "placement and every result is reported against it")
        if method not in ("exhaustive", "anneal", "surrogate"):
            raise ExperimentError(f"unknown optimization method {method!r}; use "
                                  f"'exhaustive', 'anneal' or 'surrogate'")

        pitch_um = self.initial_placement.pitch_m * 1e6
        bcs = self.config.boundary_conditions
        try:
            optimizer = ChipletPlacementOptimizer(
                nx=self.package.nx, ny=self.package.ny, nz=self.package.nz,
                dx=self.package.dx, dy=self.package.dy, dz=self.package.dz,
                catalog=tuple(c.to_spec() for c in self.catalog),
                t_ambient_c=bcs.t_ambient_c, h_convective=bcs.h_convective,
                bottom_bc=bcs.bottom_bc, top_bc=bcs.top_bc,
                stack=self.config.stack, placement_pitch_um=pitch_um,
                reuse_solver=self.solver, seed=seed)
        except OptimizerError as exc:
            raise ExperimentError(f"optimizer setup failed: {exc}") from exc

        t0 = time.perf_counter()
        if method == "exhaustive":
            try:
                result = ExhaustivePlacementSearch(
                    optimizer, placement_pitch_um=pitch_um).search(verbose=verbose)
            except (ExhaustiveSearchError, OptimizerError) as exc:
                raise ExperimentError(f"exhaustive search failed: {exc}") from exc
            best_layout = result.best_layout
            optimizer_peak = float(result.best_peak_temperature_c)
            optimizer_field = result.best_temperature_field_k
            stats = dict(candidates_evaluated=result.layouts_evaluated,
                         candidates_total=result.layouts_total,
                         rejected_or_pruned=result.layouts_total - result.layouts_evaluated,
                         precompute_solves=result.precompute_solves,
                         superposition_error_k=float(result.superposition_error_k),
                         seed=None, deterministic=True, history_c=())
        elif method == "surrogate":
            # Imported here rather than at module level: the surrogate is built
            # on this module, so importing it up top would be circular. It only
            # RANKS -- the real solver below still decides, and the re-solve
            # check still has to pass.
            from src.surrogate import SurrogateError, SurrogateScreeningSearch
            try:
                screening = SurrogateScreeningSearch(
                    self, top_k=surrogate_top_k,
                    calibration_layouts=surrogate_samples, seed=seed)
                result = screening.run(verbose=verbose)
            except SurrogateError as exc:
                raise ExperimentError(f"surrogate screening failed: {exc}") from exc
            self.surrogate = result
            best_layout = result.best_layout
            optimizer_peak = float(result.best_peak_c)
            optimizer_field = None          # the shortlist solves are not retained
            stats = dict(candidates_evaluated=result.top_k_solved,
                         candidates_total=result.candidates_in_full_space,
                         rejected_or_pruned=result.candidates_screened - result.top_k_solved,
                         precompute_solves=result.calibration_layouts,
                         superposition_error_k=None, seed=seed,
                         deterministic=False, history_c=())
        else:
            try:
                result = optimizer.optimize(
                    iterations=iterations, verbose=verbose,
                    initial_layout=self.layout_of(self.initial_placement))
            except OptimizerError as exc:
                raise ExperimentError(f"annealing failed: {exc}") from exc
            # The annealer started where we told it to; confirm it agrees
            # about that baseline, or the before/after pair is incoherent.
            if abs(result.initial_peak_temperature_c - self.initial.peak_c) > RESOLVE_TOL_K:
                raise ExperimentError(
                    f"the annealer's starting peak {result.initial_peak_temperature_c:.6f} "
                    f"degC differs from the initial solve {self.initial.peak_c:.6f} degC")
            best_layout = result.best_layout
            optimizer_peak = float(result.best_peak_temperature_c)
            optimizer_field = result.best_temperature_field_k
            stats = dict(candidates_evaluated=result.iterations, candidates_total=None,
                         rejected_or_pruned=result.overlap_candidates,
                         precompute_solves=None, superposition_error_k=None,
                         seed=result.seed, deterministic=False,
                         history_c=tuple(float(v) for v in result.best_cost_history))
        runtime_s = time.perf_counter() - t0

        optimized_placement = self.placement_of(best_layout)
        violations = validate_placement(self.package, self.catalog, optimized_placement)
        if violations:
            raise ExperimentError(
                "the optimizer returned a placement that violates its own constraints: "
                + "; ".join(violations))

        # THE AUTHORITATIVE NUMBER. The reported optimum is re-solved by the
        # real solver, never taken from a cached or superposed estimate.
        optimized = solve_placement(self.solver, self.package, self.catalog,
                                    optimized_placement, "optimized")
        delta = abs(optimized.peak_c - optimizer_peak)
        field_delta = (float(np.max(np.abs(np.asarray(optimized.field_k)
                                           - np.asarray(optimizer_field))))
                       if optimizer_field is not None else float("nan"))
        if delta > RESOLVE_TOL_K or (np.isfinite(field_delta) and field_delta > RESOLVE_TOL_K):
            raise ExperimentError(
                f"the re-solved optimum disagrees with the optimizer: re-solve "
                f"{optimized.peak_c:.6f} degC against reported {optimizer_peak:.6f} degC "
                f"(peak delta {delta:.3e} K, field delta {field_delta:.3e} K, tolerance "
                f"{RESOLVE_TOL_K:.1e} K). Refusing to report an optimum the solver does "
                f"not confirm.")

        self.optimization = OptimizationRecord(
            method=method, objective=OBJECTIVE, constraints=CONSTRAINTS,
            pitch_m=self.initial_placement.pitch_m, optimizer_peak_c=optimizer_peak,
            runtime_s=runtime_s, resolve_delta_k=delta,
            resolve_field_delta_k=field_delta, **stats)
        self.optimized = optimized
        return optimized

    # -- comparison ------------------------------------------------------
    def moves(self) -> Tuple[MoveRow, ...]:
        if self.initial is None or self.optimized is None:
            return ()
        rows: List[MoveRow] = []
        for c in self.catalog:
            x0, y0 = self.initial.placement.origin(c.id)
            x1, y1 = self.optimized.placement.origin(c.id)
            gap0, edge0 = neighbour_and_edge_gaps_m(self.package, self.catalog,
                                                    self.initial.placement, c.id)
            gap1, edge1 = neighbour_and_edge_gaps_m(self.package, self.catalog,
                                                    self.optimized.placement, c.id)
            rows.append(MoveRow(
                id=c.id, name=c.name, x0_m=x0, y0_m=y0, x1_m=x1, y1_m=y1,
                peak0_c=self.initial.chiplet(c.id).max_c,
                peak1_c=self.optimized.chiplet(c.id).max_c,
                gap0_m=gap0, gap1_m=gap1, edge0_m=edge0, edge1_m=edge1))
        return tuple(rows)

    def sanity(self, result: ThermalResult) -> Dict[str, float]:
        """The arithmetic every reported temperature should be checked
        against, computed from the same inputs the solver was given."""
        package, catalog = self.package, self.catalog
        power_w = float(sum(c.power_w for c in catalog))
        a_sink = package.sink_area_m2
        a_chip = float(sum(c.area_m2 for c in catalog))
        robin = package.sink_bc == "robin" and package.h_convective > 0
        floor = power_w / (package.h_convective * a_sink) if robin and a_sink > 0 else 0.0
        return {
            "total_power_w": power_w,
            "package_area_mm2": a_sink * 1e6,
            "chiplet_area_mm2": a_chip * 1e6,
            "fill_pct": 100.0 * a_chip / a_sink if a_sink > 0 else float("nan"),
            "flux_over_chiplets_w_cm2": power_w / (a_chip * 1e4) if a_chip > 0 else float("nan"),
            "flux_over_package_w_cm2": power_w / (a_sink * 1e4) if a_sink > 0 else float("nan"),
            "h_convective": package.h_convective,
            "t_ambient_c": package.t_ambient_c,
            "floor_rise_k": floor,
            "floor_peak_c": package.t_ambient_c + floor,
            "measured_rise_k": result.peak_c - package.t_ambient_c,
            "floor_ratio": (result.peak_c - package.t_ambient_c) / floor if floor > 0 else float("nan"),
            "specified_w": result.power.specified_w,
            "deposited_w": result.power.deposited_w,
            "sink_heat_w": result.sink_heat_w,
            "sink_heat_error_w": result.sink_heat_w - power_w,
        }

    def improvement(self) -> Dict[str, float]:
        """Peak reduction, with every denominator named. "Improvement" is a
        DIFFERENCE between two peaks and is never a rise above ambient."""
        if self.initial is None or self.optimized is None:
            return {}
        floor = self.sanity(self.initial)["floor_rise_k"]
        delta = self.initial.peak_c - self.optimized.peak_c
        rise0 = self.initial.rise_k
        excess0 = rise0 - floor
        return {
            "delta_peak_k": delta,
            "initial_peak_c": self.initial.peak_c,
            "optimized_peak_c": self.optimized.peak_c,
            "initial_rise_k": rise0,
            "optimized_rise_k": self.optimized.rise_k,
            "pct_of_initial_rise": 100.0 * delta / rise0 if rise0 > 0 else float("nan"),
            "initial_excess_over_floor_k": excess0,
            # The convective floor P/(h*A) is placement-independent, so this
            # is the fraction of the placement-addressable part removed.
            "pct_of_initial_excess": 100.0 * delta / excess0 if excess0 > 0 else float("nan"),
        }
