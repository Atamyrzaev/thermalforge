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
import logging
import math
import threading
import time
import json
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

from src.config_loader import (
    BoundaryConditionsSection, CaseSection, GridSection, PipelineConfig, PowerSection,
    SolverSection, StackSection, VisualizationSection,
)
from src.experiment import (
    ChipletDef, Placement, build_stack, package_from_config, power_z_range,
    solve_placement, validate_placement,
)
from src.surrogate import spreading_length_m
from src.thermal_solver import DIRECT_SOLVE_MAX_UNKNOWNS, ThermalSolver

for _logger in ("RISC3D-AI.ThermalSolver", "RISC3D-AI.GridBuilder",
                "RISC3D-AI.ConfigLoader", "RISC3D-AI.Optimizer", "RISC3D-AI.Exhaustive"):
    logging.getLogger(_logger).setLevel(logging.WARNING)

# Mesh sensitivity, measured by tools/mesh_convergence.py and read from its
# output. Absent that file the app simply reports no band rather than an
# invented one.
MESH_STUDY_PATH = Path(__file__).resolve().parent.parent / "validation" / "mesh_convergence.json"

NZ = 12
DZ_M = 20e-6
STACK_UM = (100.0, 40.0, 100.0)
LAYERS = (
    {"name": "Interposer (Si)", "k0": 0, "k1": 5, "thickness_um": 100.0},
    {"name": "Microbumps + underfill", "k0": 5, "k1": 7, "thickness_um": 40.0},
    {"name": "Die (Si)", "k0": 7, "k1": 12, "thickness_um": 100.0},
)
CELL_CHOICES_MM = (0.5, 1.0, 2.0)
MAX_CHIPLETS = 12

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

    @staticmethod
    def parse(raw) -> "PackageSpec":
        if not isinstance(raw, dict):
            raise StudioError("package must be an object")
        spec = PackageSpec(
            size_mm=_number(raw, "size_mm", 5.0, 40.0, "package"),
            h_w_m2k=_number(raw, "h_w_m2k", 10.0, 1e6, "package"),
            ambient_c=_number(raw, "ambient_c", -40.0, 125.0, "package"),
            pitch_mm=_number(raw, "pitch_mm", 0.5, 20.0, "package"),
        )
        cell = spec.cell_mm
        if _whole(spec.pitch_mm, cell) is None:
            raise StudioError(f"the placement grid ({spec.pitch_mm:g} mm) must be a whole "
                              f"number of {cell:g} mm mesh cells")
        return spec

    @property
    def cell_mm(self) -> float:
        """The finest lateral mesh cell that keeps the model on the direct
        (exact, factorized) solver path."""
        for cell in CELL_CHOICES_MM:
            n = _whole(self.size_mm, cell)
            if n is not None and n >= 8 and n * n * NZ <= DIRECT_SOLVE_MAX_UNKNOWNS:
                return cell
        raise StudioError(f"a {self.size_mm:g} mm package does not fit a supported mesh; "
                          f"use a whole number of millimetres")

    @property
    def n_cells(self) -> int:
        return _whole(self.size_mm, self.cell_mm)

    @property
    def model_key(self) -> Tuple[float, float, float, float]:
        return (self.size_mm, self.cell_mm, self.h_w_m2k, self.ambient_c)


@dataclass(frozen=True)
class ChipletIn:
    id: str
    name: str
    w_mm: float
    h_mm: float
    power_w: float
    x_mm: float
    y_mm: float

    def as_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "w_mm": self.w_mm, "h_mm": self.h_mm,
                "power_w": self.power_w, "x_mm": self.x_mm, "y_mm": self.y_mm}


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
        chiplet = ChipletIn(
            id=cid, name=name,
            w_mm=_number(item, "w_mm", cell, spec.size_mm, name),
            h_mm=_number(item, "h_mm", cell, spec.size_mm, name),
            power_w=_number(item, "power_w", 0.0, 2000.0, name),
            x_mm=_number(item, "x_mm", 0.0, spec.size_mm, name),
            y_mm=_number(item, "y_mm", 0.0, spec.size_mm, name),
        )
        for label, value in (("width", chiplet.w_mm), ("height", chiplet.h_mm),
                             ("x", chiplet.x_mm), ("y", chiplet.y_mm)):
            if _whole(value, cell) is None:
                raise StudioError(f"{name}: {label} {value:g} mm is not on the {cell:g} mm grid")
        out.append(chiplet)
    return tuple(out)


def _catalog(chiplets) -> Tuple[ChipletDef, ...]:
    return tuple(ChipletDef(id=c.id, name=c.name, width_um=c.w_mm * 1e3,
                            height_um=c.h_mm * 1e3, power_w=c.power_w) for c in chiplets)


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
        self._lu = sparse_linalg.splu(spd, permc_spec="MMD_AT_PLUS_A",
                                      diag_pivot_thresh=0.0,
                                      options={"SymmetricMode": True})

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """x with M x = rhs (the interface solve_steady_state() expects)."""
        rhs = np.asarray(rhs, dtype=np.float64)
        scale = self.rho_cp[:, None] if rhs.ndim == 2 else self.rho_cp
        return self._lu.solve(-(scale * rhs))

    def rise_for_power(self, q_w_m3: np.ndarray) -> np.ndarray:
        """Temperature rise theta with S theta = q, columns of q in W/m^3."""
        return self._lu.solve(np.asarray(q_w_m3, dtype=np.float64))

class ThermalModel:
    """One package: materials, boundary conditions and the factorized
    operator. Built once per (size, cell, h, ambient); every placement after
    that costs one triangular solve."""

    def __init__(self, spec: PackageSpec):
        t0 = time.perf_counter()
        n, cell_m = spec.n_cells, spec.cell_mm * 1e-3
        config = PipelineConfig(
            grid=GridSection(nx=n, ny=n, nz=NZ, dx=cell_m, dy=cell_m, dz=DZ_M),
            boundary_conditions=BoundaryConditionsSection(
                h_convective=spec.h_w_m2k, t_ambient_c=spec.ambient_c,
                bottom_bc="neumann", top_bc="robin", side_bc="neumann"),
            stack=StackSection(interposer_um=STACK_UM[0], bump_um=STACK_UM[1],
                               die_um=STACK_UM[2]),
            power=PowerSection(die_power_w=0.0, die_footprint_frac=0.5),
            solver=SolverSection(), visualization=VisualizationSection(),
            case=CaseSection(name="studio"),
        )
        config.validate()
        self.spec = spec
        self.package = package_from_config(config)
        grid_builder = build_stack(config)
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
    def __init__(self, capacity: int = 6):
        self._models: "OrderedDict[tuple, ThermalModel]" = OrderedDict()
        self._lock = threading.Lock()
        self.capacity = capacity

    def get(self, spec: PackageSpec) -> Tuple[ThermalModel, bool]:
        with self._lock:
            key = spec.model_key
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key], False
            model = ThermalModel(spec)
            self._models[key] = model
            while len(self._models) > self.capacity:
                self._models.popitem(last=False)
            return model, True


# ---------------------------------------------------------------------------
# One placement
# ---------------------------------------------------------------------------

def _load_mesh_study() -> Optional[dict]:
    try:
        return json.loads(MESH_STUDY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


_MESH_STUDY = _load_mesh_study()


def mesh_band(cell_mm: float, rise_k: float, h_w_m2k: float,
              study: Optional[dict] = None) -> Optional[dict]:
    """How much the peak still moves with the mesh, in kelvin.

    Taken from a three-mesh convergence study on a reference package (2, 1
    and 0.5 mm cells) and scaled to THIS design's rise above ambient. It is a
    discretization band, not a validation bound: it says how much of the
    answer is the mesh, and says nothing about whether the model matches
    hardware. Returns None when no study has been run.
    """
    study = _MESH_STUDY if study is None else study
    if not study or rise_k <= 0:
        return None
    cases = study.get("cases") or {}
    # The band depends on how hard the package is being cooled: under a cold
    # plate the gradients are steeper and the mesh matters more.
    name = "coldplate" if h_w_m2k >= 10000.0 and "coldplate" in cases else "passive"
    case = cases.get(name)
    if not case:
        return None
    table = case.get("band_pct_of_rise_by_cell_mm") or {}
    key = f"{cell_mm:g}"
    if key not in table:
        # Fall back to the coarsest measured mesh, which bands no tighter.
        key = max(table, key=lambda k: float(k)) if table else None
    if key is None:
        return None
    pct = float(table[key])
    return {
        "band_k": pct / 100.0 * rise_k,
        "band_pct_of_rise": pct,
        "cell_mm": float(key),
        "measured_on": name,
        "observed_order": case.get("order_p"),
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
    layer = next(l["name"] for l in LAYERS if l["k0"] <= k < l["k1"])
    by_id = {c.id: c for c in chiplets}
    rows = []
    for row in result.per_chiplet:
        c = by_id[row.id]
        area_cm2 = c.w_mm * c.h_mm / 100.0
        rows.append({"id": c.id, "name": c.name, "max_c": float(row.max_c),
                     "mean_c": float(row.mean_c), "power_w": c.power_w,
                     "density_w_cm2": c.power_w / area_cm2 if area_cm2 > 0 else 0.0})
    return {
        "grid": {"nx": pkg.nx, "ny": pkg.ny, "nz": pkg.nz, "cell_mm": spec.cell_mm,
                 "dz_um": pkg.dz * 1e6, "size_mm": spec.size_mm,
                 "layers": [dict(layer_) for layer_ in LAYERS]},
        "field_b64": base64.b64encode(field_c.astype("<f4").tobytes(order="C")).decode("ascii"),
        "t_min_c": float(field_c.min()),
        "t_max_c": float(field_c.max()),
        "ambient_c": spec.ambient_c,
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
            "mesh": mesh_band(spec.cell_mm, rise_k, spec.h_w_m2k),
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
    violations = validate_placement(model.package, catalog, placement)
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

def _check_cancel(cancel: Optional[threading.Event]) -> None:
    if cancel is not None and cancel.is_set():
        raise SearchCancelled()


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
    per_watt: Dict[Tuple[int, int], np.ndarray] = {}
    for index, (chiplet, (wc, hc), pos) in enumerate(zip(chiplets, sizes, positions)):
        _check_cancel(cancel)
        if (wc, hc) in per_watt:
            fields.append(per_watt[(wc, hc)] * chiplet.power_w)
            if on_chiplet is not None:
                on_chiplet(index + 1)
            continue
        q = 1.0 / (wc * hc * (z1 - z0) * pkg.cell_volume_m3)
        rhs = np.zeros((n_cells, len(pos)), dtype=np.float64, order="F")
        for col, (x, y) in enumerate(pos):
            block[:] = 0.0
            block[x:x + wc, y:y + hc, z0:z1] = q
            rhs[:, col] = block.reshape(n_cells)
        # S theta = q: the rise above ambient; the boundary forcing cancels.
        full = model.factor.rise_for_power(rhs)
        if full.size and float(full.min()) < -NEGATIVE_FIELD_TOL * max(1.0, float(full.max())):
            raise StudioError("a single-chiplet rise went negative; the exactness bound "
                              "would not hold, so the search refuses to run")
        theta_per_watt = np.ascontiguousarray(full[model.die_cells, :].T)
        per_watt[(wc, hc)] = theta_per_watt
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
                     improve_incumbent: bool = True,
                     cancel: Optional[threading.Event] = None,
                     on_progress: Optional[Callable[[SearchStats, float], None]] = None
                     ) -> Tuple[float, Optional[List[int]], SearchStats]:
    """Exact minimum, over every non-overlapping grid layout, of the peak rise.

    `fields[c]` is (positions, cells): chiplet c's rise at each grid position,
    over the cells where a peak can occur. Returns (best rise in K, position
    index per chiplet, stats). `incumbent` is a known layout (value, indices)
    or (inf, None); the search can only improve on it.

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
            overlap = ((np.minimum(ax + wa, bx + wb) > np.maximum(ax, bx))
                       & (np.minimum(ay + ha, by + hb) > np.maximum(ay, by)))
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

    stats = SearchStats()
    best_value, best_choice = incumbent
    chosen: List[Optional[int]] = [None] * k
    last = [time.perf_counter()]
    widest = max(len(p) for p in positions)
    sums = [np.empty((widest, nb)) for _ in range(k)]
    scratch = [np.empty((widest, nb)) for _ in range(k)]

    def descend(d: int, run_full: np.ndarray, run_part: np.ndarray) -> None:
        nonlocal best_value, best_choice
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
                on_progress(stats, best_value)

        m = int(idx.size)
        partial = sums[d][:m]
        np.take(part[ci], idx, axis=0, out=partial)
        partial += run_part

        if d == k - 1:
            stats.leaves += m
            # The subset maximum is a lower bound on each layout's peak, so only
            # layouts whose subset maximum beats the best can beat it in full.
            contenders = idx[partial.max(axis=1) < best_value]
            if contenders.size:
                peaks = (full[ci][contenders] + run_full).max(axis=1)
                j = int(np.argmin(peaks))
                if peaks[j] < best_value:
                    best_value = float(peaks[j])
                    chosen[ci] = int(contenders[j])
                    best_choice = list(chosen)
            chosen[ci] = None
            return

        bound_rows = scratch[d][:m]
        np.add(partial, rest[d + 1], out=bound_rows)
        bound = bound_rows.max(axis=1)
        ranked = np.argsort(bound, kind="stable")
        for rank, j in enumerate(ranked):
            if bound[j] >= best_value:
                stats.pruned_bound += int(ranked.size - rank)
                break
            chosen[ci] = int(idx[j])
            descend(d + 1, run_full + full[ci][idx[j]], partial[j])
        chosen[ci] = None

    descend(0, np.zeros(n_cells), np.zeros(nb))
    if on_progress is not None:
        on_progress(stats, best_value)
    return best_value, best_choice, stats


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
    violations = validate_placement(pkg, catalog, _placement(chiplets, spec.cell_mm))
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

    signatures = [(sizes[i], chiplets[i].power_w) for i in range(len(chiplets))]

    t0 = time.perf_counter()

    def search_progress(stats: SearchStats, best: float) -> None:
        report("searching every layout", nodes=stats.nodes, leaves=stats.leaves,
               best_c=(best + spec.ambient_c) if math.isfinite(best) else None,
               combinations=combinations)

    best_value, best_choice, stats = branch_and_bound(
        fields, positions, sizes, signatures, (incumbent_value, incumbent_choice),
        bound_cells=model.bound_cells, cancel=cancel, on_progress=search_progress)
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
    winner_violations = validate_placement(pkg, catalog, _placement(winners, cell))
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
