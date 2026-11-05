"""
ai/rom.py
ThermalForge :: a reduced-order plate-stack model -- the surrogate's physics.

A learned model that has to predict a package it has never seen needs the
physics of that package from somewhere, and a curve fit over a few features
does not carry it: change the cooling stack and every coefficient changes
meaning. So the surrogate starts from physics that DOES carry across packages
and learns only what that physics leaves out.

The physics: every material layer of the package -- interposer, bumps (with a
BEOL folded in if the design has one), the die, and each layer of the cooling
path -- is one plate, one node thick. The die's device plane, where the engine
deposits power, is a plate of its own at the bumped face. Plates conduct
sideways with their sheet conductance k*t and exchange heat vertically through
the series resistance between their mid-planes; the top plate meets the
coolant through h; the bottom and the sides are adiabatic, as in the engine.
Materials, thicknesses and options are read from the engine's own PackageSpec,
so the two models cannot describe different packages.

What it gives up, on purpose: vertical resolution inside a layer, and -- at a
coarse lateral cell -- the hotspot inside a patterned part. Those are exactly
what the learned correction (ai.surrogate) is trained to put back, from
features that describe them. What it keeps: how much heat each cooling stack
spreads and how far, which is what changes from package to package.

Cost: a 40 mm package at 2 mm cells with a lid and heatsink is 20 x 20 x 8 =
3,200 unknowns -- a factorization in milliseconds and a solve in a fraction of
one -- against 19,200-77,000 for the engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from src.composite_materials import BEOL_COPPER_FRACTION, beol_bump_layer
from src.grid_builder import MATERIALS

DEVICE_UM = 20.0            # the engine's device layer: one 20 um cell at the bumped face


@dataclass(frozen=True)
class Plate:
    name: str
    t_m: float
    kxy: float
    kz: float


def plates(spec) -> List[Plate]:
    """The package as plates, bottom to top, from the engine's own spec."""
    t_i, t_b, t_d = (t * 1e-6 for t in spec.stack_um)
    if spec.interposer_material == "silicon":
        k_si = float(MATERIALS["silicon"]["k"])
        inter = Plate("interposer", t_i, k_si, k_si)
    else:
        m = MATERIALS[spec.interposer_material]
        inter = Plate("interposer", t_i, float(m["kx"]), float(m["kz"]))
    bump = dict(MATERIALS["bump_underfill_composite"])
    if getattr(spec, "beol_um", 0.0) > 0:
        bump = beol_bump_layer(bump, spec.stack_um[1], spec.beol_um, BEOL_COPPER_FRACTION)
    bumps = Plate("bumps", t_b, float(bump["kx"]), float(bump["kz"]))
    k_si = float(MATERIALS["silicon"]["k"])
    out = [inter, bumps, Plate("device", DEVICE_UM * 1e-6, k_si, k_si)]
    rest = t_d - DEVICE_UM * 1e-6
    if rest > 1e-9:
        out.append(Plate("die", rest, k_si, k_si))
    for name, thickness_um, k, _cells in spec.cooling_layers:
        out.append(Plate(str(name), thickness_um * 1e-6, float(k), float(k)))
    return out


def spreading_length_m(spec) -> float:
    """sqrt(sheet conductance / (conductance to the coolant per area)), with
    the device plane as the source: how far heat travels sideways."""
    ps = plates(spec)
    sheet = sum(p.kxy * p.t_m for p in ps)
    dev = next(i for i, p in enumerate(ps) if p.name == "device")
    r = sum(p.t_m / p.kz for p in ps[dev + 1:]) + 0.5 * ps[dev].t_m / ps[dev].kz + 1.0 / spec.h_w_m2k
    return math.sqrt(sheet * r)


def _overlap_1d(n: int, cell: float, a: float, b: float) -> np.ndarray:
    """Length of [a, b) inside each of n cells of width `cell` starting at 0."""
    edges = np.arange(n + 1) * cell
    return np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0.0, None)


def part_power_map(part, nx: int, ny: int, cell_mm: float) -> np.ndarray:
    """Watts per ROM cell for one parsed chiplet (studio.engine.ChipletIn),
    deposited by exact area overlap -- uniform share, blocks and imported
    tiles alike -- so any cell size conserves the part's power."""
    q = np.zeros((nx, ny))
    x0, y0, w, h = part.x_mm, part.y_mm, part.w_mm, part.h_mm

    def add(ax, ay, bx, by, watts):
        ox = _overlap_1d(nx, cell_mm, ax, bx)
        oy = _overlap_1d(ny, cell_mm, ay, by)
        area = float(ox.sum() * oy.sum())
        if area > 0:
            q[:, :] += watts * np.outer(ox, oy) / area

    grid = getattr(part, "power_grid", None)
    if grid is not None and grid.cols and grid.rows:
        wts = np.asarray(grid.weights, dtype=float).reshape(grid.rows, grid.cols)
        total = float(wts.sum()) or 1.0
        tw, th = w / grid.cols, h / grid.rows
        for r in range(grid.rows):
            for c in range(grid.cols):
                if wts[r, c] > 0:
                    # row 0 is the top of the part, as in the engine's reader
                    ay = y0 + (grid.rows - 1 - r) * th
                    add(x0 + c * tw, ay, x0 + (c + 1) * tw, ay + th, part.power_w * wts[r, c] / total)
        return q
    uniform = 1.0 - sum(b.share for b in part.blocks)
    if uniform > 1e-12:
        add(x0, y0, x0 + w, y0 + h, part.power_w * uniform)
    for b in part.blocks:
        add(x0 + b.x_frac * w, y0 + b.y_frac * h, x0 + (b.x_frac + b.w_frac) * w,
            y0 + (b.y_frac + b.h_frac) * h, part.power_w * b.share)
    return q


def footprint_mask(part, nx: int, ny: int, cell_mm: float) -> np.ndarray:
    ox = _overlap_1d(nx, cell_mm, part.x_mm, part.x_mm + part.w_mm)
    oy = _overlap_1d(ny, cell_mm, part.y_mm, part.y_mm + part.h_mm)
    return np.outer(ox > 1e-9, oy > 1e-9)


class PlateStack:
    """The factorized reduced-order model of one package."""

    def __init__(self, spec, cell_mm: float) -> None:
        self.spec = spec
        self.cell_mm = float(cell_mm)
        self.nx = max(1, int(round(spec.size_mm / cell_mm)))
        self.ny = max(1, int(round(spec.size_y / cell_mm)))
        self.plates = plates(spec)
        self.device = next(i for i, p in enumerate(self.plates) if p.name == "device")
        nl, nx, ny = len(self.plates), self.nx, self.ny
        n = nl * nx * ny
        dx = self.cell_mm * 1e-3
        area = dx * dx
        idx = np.arange(n).reshape(nl, nx, ny)
        rows, cols, vals = [], [], []
        diag = np.zeros(n)

        def couple(a: np.ndarray, b: np.ndarray, g) -> None:
            a, b = a.ravel(), b.ravel()
            g = np.broadcast_to(g, a.shape).ravel()
            rows.extend((a, b)); cols.extend((b, a)); vals.extend((-g, -g))
            np.add.at(diag, a, g)
            np.add.at(diag, b, g)

        for L, p in enumerate(self.plates):
            g_lat = p.kxy * p.t_m                  # k t dy/dx with dx = dy
            if nx > 1:
                couple(idx[L, :-1, :], idx[L, 1:, :], g_lat)
            if ny > 1:
                couple(idx[L, :, :-1], idx[L, :, 1:], g_lat)
            if L + 1 < nl:
                q = self.plates[L + 1]
                g_v = area / (0.5 * p.t_m / p.kz + 0.5 * q.t_m / q.kz)
                couple(idx[L], idx[L + 1], g_v)
        top = self.plates[-1]
        g_top = area / (0.5 * top.t_m / top.kz + 1.0 / spec.h_w_m2k)
        diag[idx[-1].ravel()] += g_top
        m = sp.coo_matrix((np.concatenate(vals) if vals else np.zeros(0),
                           (np.concatenate(rows) if rows else np.zeros(0, int),
                            np.concatenate(cols) if cols else np.zeros(0, int))), shape=(n, n))
        self.matrix = (m + sp.diags(diag)).tocsc()
        self._lu = spla.splu(self.matrix)
        self.g_top_total = g_top * nx * ny
        self.unknowns = n

    def rise(self, q_device: np.ndarray) -> np.ndarray:
        """Rise above ambient (nl, nx, ny) for watts per cell on the device plane."""
        rhs = np.zeros((len(self.plates), self.nx, self.ny))
        rhs[self.device] = q_device
        return self._lu.solve(rhs.ravel()).reshape(len(self.plates), self.nx, self.ny)

    def device_rise(self, q_device: np.ndarray) -> np.ndarray:
        return self.rise(q_device)[self.device]


def rom_cell_mm(spec, factor: float = 2.0, min_cell_mm: float = 0.25) -> float:
    """A cell that divides the package: `factor` times the engine's standard
    cell, so the ROM is always coarser than what it is corrected against."""
    cell = max(min_cell_mm, factor * spec.cell_mm)
    n = max(1, int(round(spec.size_mm / cell)))
    return spec.size_mm / n


class RomCache:
    """Factorized plate stacks by package, least recently used out."""

    def __init__(self, capacity: int = 16, factor: float = 2.0) -> None:
        self.capacity = capacity
        self.factor = factor
        self._items: Dict[tuple, PlateStack] = {}

    def get(self, spec) -> PlateStack:
        cell = rom_cell_mm(spec, self.factor)
        key = (spec.model_key, round(cell, 9))
        if key in self._items:
            stack = self._items.pop(key)
        else:
            stack = PlateStack(spec, cell)
            while len(self._items) >= self.capacity:
                self._items.pop(next(iter(self._items)))
        self._items[key] = stack
        return stack


@dataclass
class RomSolve:
    stack: PlateStack
    device: np.ndarray                  # device-plane rise, (nx, ny)
    part_rise: List[float]              # max rise over each part's footprint
    self_rise: List[float]              # the same, from that part's own power alone
    part_q: List[np.ndarray]


def solve_layout(cache: RomCache, spec, chiplets: Sequence) -> RomSolve:
    """ROM rise for a parsed layout, with each part's own contribution split
    out (by superposition: one extra back-substitution per part)."""
    stack = cache.get(spec)
    qs = [part_power_map(c, stack.nx, stack.ny, stack.cell_mm) for c in chiplets]
    selfs = [stack.device_rise(q) for q in qs]
    total = np.sum(selfs, axis=0) if selfs else np.zeros((stack.nx, stack.ny))
    masks = [footprint_mask(c, stack.nx, stack.ny, stack.cell_mm) for c in chiplets]
    part_rise = [float(total[m].max()) if m.any() else 0.0 for m in masks]
    self_rise = [float(s[m].max()) if m.any() else 0.0 for s, m in zip(selfs, masks)]
    return RomSolve(stack=stack, device=total, part_rise=part_rise, self_rise=self_rise, part_q=qs)
