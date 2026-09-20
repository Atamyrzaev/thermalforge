"""
grid_builder.py
RISC3D-AI :: 3D IC / Advanced Packaging Thermal Modeling Framework

GridBuilder discretizes heterogeneous 3D chip/package geometries (TSMC CoWoS-S,
SoIC, InFO stacks) into voxelized NumPy arrays carrying per-cell material
properties (thermal conductivity tensor, density, specific heat) and power
density (Q) for consumption by ThermalSolver.

Design notes:
- Conductivity is stored as three separate arrays (kx_map, ky_map, kz_map)
  rather than a single scalar k_map, because several layers in the stack
  (microbump+underfill composite, in particular) are strongly anisotropic:
  vertical (z) conduction runs up the bump columns while lateral (xy)
  conduction crosses the low-k underfill -- 44x apart for the derived
  composite (kz 13.148 vs kxy 0.2975 W/mK, src/composite_materials.py).
- All spatial assignment methods operate on index slices, not physical
  coordinates, to keep the hot path allocation-free and vectorized.
- Every public mutating method validates shapes/bounds before writing, and
  raises informative exceptions rather than silently corrupting the grid.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np

from src.composite_materials import (CITATION as TSCI_CITATION, TSCI_K, beol_k,
                                     bump_underfill_layer_k, tsv_cell_k)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.GridBuilder")


# ---------------------------------------------------------------------------
# Material property database
# k in W/(m K), rho in kg/m^3, cp in J/(kg K). Anisotropic entries carry
# (kx, ky, kz); isotropic entries carry one k broadcast to all three axes.
#
# Conductivities are Table 2 of TSCI_CITATION, or derived from it in
# src/composite_materials.py. Old -> new, replacing values that had no source:
#   silicon 149 -> 150, copper 400 -> 390, underfill 0.8 -> 0.2,
#   solder 60 -> 57 (Sn2.5Ag), composite kx=ky 2.0 -> 0.2975, kz 20.0 -> 13.148.
# rho and cp are NOT from that paper (a steady-state study; it gives none).
# They are the earlier, uncited values and affect only transients and tau --
# never a steady-state temperature. Entries marked UNCITED have no source.
# ---------------------------------------------------------------------------

_COMPOSITE_KX, _COMPOSITE_KY, _COMPOSITE_KZ = bump_underfill_layer_k()
# The paper's own TSV cell: 8 um copper core, 9 um to the outside of the
# SiO2 liner, 40 um pitch.
_TSV_KXY, _TSV_KZ = tsv_cell_k(8.0, 9.0, 40.0)
_BEOL_KXY, _BEOL_KZ = beol_k()

MATERIALS: Dict[str, Dict[str, float]] = {
    "silicon": {"k": TSCI_K["silicon"], "rho": 2330.0, "cp": 712.0},
    "copper": {"k": TSCI_K["copper"], "rho": 8960.0, "cp": 385.0},
    # UNCITED: TSCI220805061R gives no TIM value. Used only by
    # build_cowos_s_reference_stack() below, never by the main.py pipeline.
    "tim": {"k": 12.5, "rho": 2000.0, "cp": 1000.0},
    "solder_microbump": {"k": TSCI_K["sn2.5ag"], "rho": 7300.0, "cp": 230.0},   # Sn2.5Ag
    "underfill": {"k": TSCI_K["underfill"], "rho": 1200.0, "cp": 1100.0},
    # Microbump array in underfill: Table 1 bump laminate through eqs.
    # (10)-(11), then eq. (2) along the pillars and Maxwell-Garnett across
    # them, at an assumed 40 um pitch. Derivation, validation against the
    # paper's own numbers, and assumptions: src/composite_materials.py.
    "bump_underfill_composite": {
        "kx": _COMPOSITE_KX, "ky": _COMPOSITE_KY, "kz": _COMPOSITE_KZ,
        "rho": 3500.0,                          # UNCITED volumetric estimate
        "cp": 450.0,                            # UNCITED
    },
    # Silicon carrying TSVs: a copper core in an SiO2 liner on a square
    # pitch, eq. (2) along the vias and Maxwell-Garnett across them. Checked
    # against the paper's own cell (r 8 / 9 um, 40 um pitch): this gives
    # kxy 114.4 against its FEM 115.3, and kz 175.2 against its 175.2.
    # Derivation in src/composite_materials.py (tsv_cell_k).
    "tsv_interposer": {
        "kx": _TSV_KXY, "ky": _TSV_KXY, "kz": _TSV_KZ,
        "rho": 2330.0,                          # silicon dominates the volume
        "cp": 712.0,
    },
    # Copper/SiO2 interconnect on the active face, eqs. (10)-(11). About
    # 2 W/(m K) through and over 100 along -- see composite_materials.beol_k.
    # NOT in any stack this codebase builds: a real BEOL is thinner than one
    # 20 um z-cell and needs the compact-layer treatment first.
    "beol": {
        "kx": _BEOL_KXY, "ky": _BEOL_KXY, "kz": _BEOL_KZ,
        "rho": 2800.0,                          # UNCITED volumetric estimate
        "cp": 700.0,                            # UNCITED
    },
    # UNCITED textbook value; not used by any stack in this codebase.
    "ambient_air": {"k": 0.026, "rho": 1.2, "cp": 1005.0},
}

# Recommended spatial discretization bounds from the physical spec.
DX_DY_MIN, DX_DY_MAX = 10e-6, 50e-6
DZ_MIN, DZ_MAX = 5e-6, 20e-6

BoundaryType = str  # one of: "dirichlet", "robin", "neumann"


@dataclass
class GridConfig:
    """Resolved, validated grid configuration."""
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float
    t_ambient_k: float = 298.15          # 25 degC in Kelvin
    # Defensive fallback, reachable only via direct GridBuilder(config=...); main.py always passes h.
    h_convective: float = 2500.0          # W/m^2K, midpoint of 1000-5000
    bottom_bc: BoundaryType = "neumann"
    top_bc: BoundaryType = "robin"
    side_bc: BoundaryType = "neumann"

    @property
    def sink_face(self) -> str:
        """'top' or 'bottom' -- whichever face carries the non-adiabatic BC."""
        return "bottom" if self.bottom_bc != "neumann" else "top"

    @property
    def sink_bc(self) -> str:
        """'robin' or 'dirichlet' -- the kind of sink on the sink face."""
        return self.bottom_bc if self.bottom_bc != "neumann" else self.top_bc

    def validate(self) -> None:
        if min(self.nx, self.ny, self.nz) <= 0:
            raise ValueError(f"Grid dimensions must be positive, got "
                              f"({self.nx}, {self.ny}, {self.nz})")
        # SCALE-AWARE resolution check.
        #
        # This used to compare dx/dy against an absolute 10-50 um band and
        # dz against 5-20 um. Those numbers were written for a die-scale mesh
        # and are simply wrong at package scale: a 20 mm module over 40 voxels
        # needs dx = 500 um, which is correct and which the old check flagged
        # on every single run. A warning that is always wrong is worse than no
        # warning, because it teaches the user to ignore the channel.
        #
        # What actually matters is not the absolute cell size but whether the
        # mesh RESOLVES the domain: too few cells per axis cannot represent
        # any internal structure, whatever the units say.
        min_cells_per_axis = 8
        if min(self.nx, self.ny, self.nz) < min_cells_per_axis:
            logger.warning(
                "Grid is %dx%dx%d; fewer than %d cells on an axis cannot resolve internal "
                "structure (layers, block edges, gradients) regardless of cell size.",
                self.nx, self.ny, self.nz, min_cells_per_axis,
            )

        # Extreme cell aspect ratios degrade iterative-solver conditioning.
        # Reported, not forbidden: the package-scale default is 25:1 and
        # solves cleanly on the direct and CG paths alike.
        aspect = max(self.dx, self.dy) / self.dz if self.dz > 0 else float("inf")
        if aspect > 100.0:
            logger.warning(
                "Cell aspect ratio is %.0f:1 (dx=%.3e, dz=%.3e). Above ~100:1 the "
                "steady-state operator becomes poorly conditioned and iterative solves "
                "may need many more iterations.", aspect, self.dx, self.dz,
            )
        valid = ("dirichlet", "robin", "neumann")
        for name, value in (("bottom_bc", self.bottom_bc), ("top_bc", self.top_bc)):
            if value not in valid:
                raise ValueError(f"{name} must be one of {valid}, got {value}")
        sinks = [v for v in (self.bottom_bc, self.top_bc) if v != "neumann"]
        if len(sinks) != 1:
            raise ValueError(
                f"Exactly one of bottom_bc / top_bc must be the heat sink; got "
                f"bottom_bc={self.bottom_bc}, top_bc={self.top_bc}")
        if self.side_bc != "neumann":
            logger.warning("Side BC is non-adiabatic (%s); confirm this is intentional.",
                           self.side_bc)


class GridBuilderError(Exception):
    """Raised on invalid geometry assignment, config, or bounds violations."""


class GridBuilder:
    """
    Constructs and manages the voxelized 3D material/power grid for a
    TSMC-style advanced packaging thermal stack (CoWoS-S / SoIC / InFO).

    Arrays are indexed [ix, iy, iz] with ix in [0, nx), iy in [0, ny),
    iz in [0, nz). iz=0 is the bottom of the stack (interposer underside);
    iz=nz-1 is the topmost layer (die backside). Which of the two faces is
    the heat sink is a boundary condition, not an indexing convention -- see
    GridConfig.sink_face; the default is the top.
    """

    DEFAULT_CONFIG_PATH = os.path.join("config", "simulation_config.json")

    def __init__(self, config: Optional[Union[str, Dict]] = None):
        """
        Parameters
        ----------
        config : str | dict | None
            Path to a simulation_config.json, a pre-loaded dict, or None to
            fall back to DEFAULT_CONFIG_PATH (with a hardcoded safe default
            if that file is also unavailable).
        """
        self.config: GridConfig = self._load_config(config)
        self.config.validate()

        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz
        shape = (nx, ny, nz)

        # Anisotropic conductivity tensor components (W/m.K)
        self.kx_map = np.zeros(shape, dtype=np.float64)
        self.ky_map = np.zeros(shape, dtype=np.float64)
        self.kz_map = np.zeros(shape, dtype=np.float64)

        # Density (kg/m^3) and specific heat (J/kg.K)
        self.rho_map = np.zeros(shape, dtype=np.float64)
        self.cp_map = np.zeros(shape, dtype=np.float64)

        # Volumetric power generation density (W/m^3)
        self.q_map = np.zeros(shape, dtype=np.float64)

        # Boolean mask: True where a voxel has been assigned a material.
        # Used to detect un-filled ("vacuum") cells before solve.
        self._assigned_mask = np.zeros(shape, dtype=bool)

        # Boundary-face tags per voxel, populated by tag_boundaries().
        # bit0=bottom(-z), bit1=top(+z), bit2=-x, bit3=+x, bit4=-y, bit5=+y
        self.boundary_mask = np.zeros(shape, dtype=np.uint8)

        logger.info(
            "GridBuilder initialized: shape=(%d, %d, %d), dx=%.2eum dy=%.2eum dz=%.2eum",
            nx, ny, nz, self.config.dx * 1e6, self.config.dy * 1e6, self.config.dz * 1e6,
        )

    # ------------------------------------------------------------------
    # Config loading
    # ------------------------------------------------------------------
    # This loader and its .get() defaults exist for DIRECT GridBuilder(config=...)
    # construction only. The pipeline passes PipelineConfig.to_dict(), which always
    # carries every grid and boundary key (test_main_always_supplies_every_grid_key).
    def _load_config(self, config: Optional[Union[str, Dict]]) -> GridConfig:
        raw: Dict
        try:
            if config is None:
                if os.path.exists(self.DEFAULT_CONFIG_PATH):
                    with open(self.DEFAULT_CONFIG_PATH, "r") as f:
                        raw = json.load(f)
                    logger.info("Loaded config from %s", self.DEFAULT_CONFIG_PATH)
                else:
                    logger.warning(
                        "No config path given and %s not found; using hardcoded "
                        "fallback grid config.", self.DEFAULT_CONFIG_PATH
                    )
                    raw = {}
            elif isinstance(config, str):
                with open(config, "r") as f:
                    raw = json.load(f)
                logger.info("Loaded config from %s", config)
            elif isinstance(config, dict):
                raw = config
            else:
                raise TypeError(f"config must be str, dict, or None, got {type(config)}")
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to load config (%s); falling back to defaults.", exc)
            raw = {}

        grid_section = raw.get("grid", {})
        bc_section = raw.get("boundary_conditions", {})

        return GridConfig(
            nx=int(grid_section.get("nx", 200)),
            ny=int(grid_section.get("ny", 200)),
            nz=int(grid_section.get("nz", 60)),
            dx=float(grid_section.get("dx", 25e-6)),
            dy=float(grid_section.get("dy", 25e-6)),
            dz=float(grid_section.get("dz", 10e-6)),
            t_ambient_k=float(bc_section.get("t_ambient_k", 298.15)),
            h_convective=float(bc_section.get("h_convective", 2500.0)),
            bottom_bc=bc_section.get("bottom_bc", "neumann"),
            top_bc=bc_section.get("top_bc", "robin"),
            side_bc=bc_section.get("side_bc", "neumann"),
        )

    # ------------------------------------------------------------------
    # Bounds / shape validation helpers
    # ------------------------------------------------------------------
    def _validate_slice_bounds(
        self, x_slice: slice, y_slice: slice, z_slice: slice
    ) -> Tuple[slice, slice, slice]:
        nx, ny, nz = self.config.nx, self.config.ny, self.config.nz

        def _check(s: slice, dim_size: int, name: str) -> slice:
            start = 0 if s.start is None else s.start
            stop = dim_size if s.stop is None else s.stop
            if start < 0 or stop > dim_size or start >= stop:
                raise GridBuilderError(
                    f"Invalid {name}-slice ({start}:{stop}) for dimension size {dim_size}"
                )
            return slice(start, stop)

        return (
            _check(x_slice, nx, "x"),
            _check(y_slice, ny, "y"),
            _check(z_slice, nz, "z"),
        )

    @staticmethod
    def _resolve_material(material: str) -> Dict[str, float]:
        if material not in MATERIALS:
            raise GridBuilderError(
                f"Unknown material '{material}'. Available: {sorted(MATERIALS.keys())}"
            )
        return MATERIALS[material]

    # ------------------------------------------------------------------
    # Material assignment
    # ------------------------------------------------------------------
    def assign_region(
        self,
        material: str,
        x_slice: slice = slice(None),
        y_slice: slice = slice(None),
        z_slice: slice = slice(None),
        overrides: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Assign a homogeneous (or overridden) material to a rectangular voxel
        region. Handles both isotropic materials (single 'k') and anisotropic
        materials (kx/ky/kz), e.g. 'bump_underfill_composite'.

        Parameters
        ----------
        material : str
            Key into MATERIALS.
        x_slice, y_slice, z_slice : slice
            Index-space region to fill. Defaults to full extent on that axis.
        overrides : dict, optional
            Per-call property overrides, e.g. {"k": 8.0} to set a specific
            TIM conductivity within its 5-20 W/mK spec range, or
            {"kz": 22.0, "kx": 2.5, "ky": 2.5} for the composite layer.
        """
        try:
            xs, ys, zs = self._validate_slice_bounds(x_slice, y_slice, z_slice)
            props = dict(self._resolve_material(material))
            if overrides:
                props.update(overrides)

            if "k" in props:
                kx = ky = kz = float(props["k"])
            else:
                required = {"kx", "ky", "kz"}
                missing = required - props.keys()
                if missing:
                    raise GridBuilderError(
                        f"Material '{material}' missing anisotropic keys: {missing}"
                    )
                kx, ky, kz = float(props["kx"]), float(props["ky"]), float(props["kz"])

            if "rho" not in props or "cp" not in props:
                raise GridBuilderError(f"Material '{material}' missing rho/cp definition.")

            self.kx_map[xs, ys, zs] = kx
            self.ky_map[xs, ys, zs] = ky
            self.kz_map[xs, ys, zs] = kz
            self.rho_map[xs, ys, zs] = float(props["rho"])
            self.cp_map[xs, ys, zs] = float(props["cp"])
            self._assigned_mask[xs, ys, zs] = True

            logger.info(
                "Assigned '%s' to region x[%s:%s] y[%s:%s] z[%s:%s] "
                "(kx=%.2f ky=%.2f kz=%.2f W/mK)",
                material, xs.start, xs.stop, ys.start, ys.stop, zs.start, zs.stop,
                kx, ky, kz,
            )
        except GridBuilderError:
            raise
        except Exception as exc:  # noqa: BLE001 - convert any unexpected failure
            raise GridBuilderError(f"assign_region failed for material '{material}': {exc}") from exc

    def assign_layer(self, z_index: Union[int, slice], material: str,
                      overrides: Optional[Dict[str, float]] = None) -> None:
        """Convenience wrapper: assign a material across the full x,y extent
        at a given z-layer (or z-range), matching the CoWoS-S stack layers
        (heat sink / TIM / chiplets / microbump+underfill / interposer /
        substrate)."""
        z_slice = z_index if isinstance(z_index, slice) else slice(z_index, z_index + 1)
        self.assign_region(material, slice(None), slice(None), z_slice, overrides)

    def add_power_source(
        self,
        power_watts: float,
        x_slice: slice,
        y_slice: slice,
        z_slice: slice,
        additive: bool = True,
    ) -> None:
        """
        Distribute a lumped power value (W) uniformly across the voxel volume
        of the given region, converting to volumetric power density Q (W/m^3).

        Parameters
        ----------
        power_watts : float
            Total power dissipated by this die/region (e.g. logic compute
            die or HBM stack), must be >= 0.
        additive : bool
            If True, adds to existing Q in the region (for overlapping/
            multi-source superposition). If False, overwrites.
        """
        if power_watts < 0:
            raise GridBuilderError(f"power_watts must be >= 0, got {power_watts}")

        try:
            xs, ys, zs = self._validate_slice_bounds(x_slice, y_slice, z_slice)
            n_cells = (xs.stop - xs.start) * (ys.stop - ys.start) * (zs.stop - zs.start)
            if n_cells <= 0:
                raise GridBuilderError("Power source region has zero volume.")

            cell_volume = self.config.dx * self.config.dy * self.config.dz
            region_volume = n_cells * cell_volume
            if region_volume <= 0:
                raise GridBuilderError("Computed region volume is non-positive.")

            q_density = power_watts / region_volume  # W/m^3

            if additive:
                self.q_map[xs, ys, zs] += q_density
            else:
                self.q_map[xs, ys, zs] = q_density

            logger.info(
                "Applied power source: %.4f W over %d cells (region_vol=%.3e m^3) "
                "-> Q=%.4e W/m^3 (%s)",
                power_watts, n_cells, region_volume, q_density,
                "additive" if additive else "overwrite",
            )
        except GridBuilderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GridBuilderError(f"add_power_source failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Boundary tagging
    # ------------------------------------------------------------------
    def tag_boundaries(self) -> None:
        """
        Populate boundary_mask with face flags for voxels adjacent to the
        6 outer faces of the domain. These bits are pure GEOMETRY -- which
        face a voxel touches -- and say nothing about which face is the heat
        sink. That is set by GridConfig.bottom_bc / top_bc (see sink_face);
        the default is the TOP face, the die backside.
        """
        self.boundary_mask[:, :, :] = 0
        self.boundary_mask[:, :, 0] |= 0b000001         # bottom (-z) face
        self.boundary_mask[:, :, -1] |= 0b000010        # top (+z) face
        self.boundary_mask[0, :, :] |= 0b000100         # -x side
        self.boundary_mask[-1, :, :] |= 0b001000        # +x side
        self.boundary_mask[:, 0, :] |= 0b010000         # -y side
        self.boundary_mask[:, -1, :] |= 0b100000        # +y side
        logger.info("Boundary mask tagged for all 6 domain faces.")

    # ------------------------------------------------------------------
    # Validation / diagnostics
    # ------------------------------------------------------------------
    def validate_grid(self, raise_on_unassigned: bool = True) -> None:
        """
        Sanity-check the fully assembled grid prior to handing off to
        ThermalSolver: no unfilled voxels, no non-physical (<=0) k/rho/cp,
        no NaN/Inf contamination.
        """
        unassigned = int(np.count_nonzero(~self._assigned_mask))
        if unassigned > 0:
            msg = f"{unassigned} voxels have no assigned material."
            if raise_on_unassigned:
                raise GridBuilderError(msg)
            logger.warning(msg)

        for name, arr in (
            ("kx_map", self.kx_map), ("ky_map", self.ky_map), ("kz_map", self.kz_map),
            ("rho_map", self.rho_map), ("cp_map", self.cp_map),
        ):
            if not np.all(np.isfinite(arr)):
                raise GridBuilderError(f"{name} contains NaN/Inf values.")
            assigned_vals = arr[self._assigned_mask]
            if assigned_vals.size and np.any(assigned_vals <= 0):
                raise GridBuilderError(f"{name} contains non-physical (<=0) values in assigned cells.")

        if not np.all(np.isfinite(self.q_map)):
            raise GridBuilderError("q_map contains NaN/Inf values.")

        logger.info("Grid validation passed: shape=%s, unassigned_cells=%d",
                    self.kx_map.shape, unassigned)

    def get_max_diffusivity_per_axis(self) -> Dict[str, float]:
        """
        Returns max(k_axis / (rho*cp)) per axis across assigned cells — the
        quantity ThermalSolver needs (combined with dx/dy/dz) to compute the
        per-cell CFL-limited explicit timestep bound:

            dt <= 1 / [2*(alpha_x/dx^2 + alpha_y/dy^2 + alpha_z/dz^2)]
        """
        mask = self._assigned_mask & (self.rho_map > 0) & (self.cp_map > 0)
        if not np.any(mask):
            raise GridBuilderError("No assigned cells with valid rho/cp; cannot compute diffusivity.")
        denom = self.rho_map[mask] * self.cp_map[mask]
        return {
            "alpha_x_max": float(np.max(self.kx_map[mask] / denom)),
            "alpha_y_max": float(np.max(self.ky_map[mask] / denom)),
            "alpha_z_max": float(np.max(self.kz_map[mask] / denom)),
        }

    def compute_cfl_dt(self, safety_factor: float = 0.9) -> float:
        """
        Computes the maximum stable explicit-FDM timestep (seconds) given the
        current material assignment and grid spacing, per the anisotropic
        7-point-stencil CFL bound, scaled by a safety factor (<1.0) for
        numerical margin.
        """
        if not (0.0 < safety_factor <= 1.0):
            raise GridBuilderError(f"safety_factor must be in (0, 1], got {safety_factor}")
        alpha = self.get_max_diffusivity_per_axis()
        dx2, dy2, dz2 = self.config.dx ** 2, self.config.dy ** 2, self.config.dz ** 2
        denom = 2.0 * (
            alpha["alpha_x_max"] / dx2 + alpha["alpha_y_max"] / dy2 + alpha["alpha_z_max"] / dz2
        )
        if denom <= 0:
            raise GridBuilderError("Non-positive CFL denominator; check material/grid assignment.")
        dt_max = 1.0 / denom
        dt_safe = dt_max * safety_factor
        logger.info("CFL dt_max=%.4e s, safety_factor=%.2f -> dt_used=%.4e s",
                    dt_max, safety_factor, dt_safe)
        return dt_safe

    def hotspot_candidates_mask(self, temperature_field: np.ndarray, threshold_c: float = 85.0) -> np.ndarray:
        """
        Given a T[x,y,z] array in degrees C (same shape as internal grids),
        returns a boolean mask of cells exceeding threshold_c, per the
        EDA hotspot-detection requirement.
        """
        if temperature_field.shape != self.kx_map.shape:
            raise GridBuilderError(
                f"temperature_field shape {temperature_field.shape} does not match "
                f"grid shape {self.kx_map.shape}"
            )
        if not np.all(np.isfinite(temperature_field)):
            raise GridBuilderError("temperature_field contains NaN/Inf values.")
        return temperature_field > threshold_c

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_arrays(self) -> Dict[str, np.ndarray]:
        """Bundle all grid arrays for handoff to ThermalSolver."""
        return {
            "kx_map": self.kx_map,
            "ky_map": self.ky_map,
            "kz_map": self.kz_map,
            "rho_map": self.rho_map,
            "cp_map": self.cp_map,
            "q_map": self.q_map,
            "boundary_mask": self.boundary_mask,
            "assigned_mask": self._assigned_mask,
        }

    def save_npz(self, filepath: str) -> None:
        """Persist the assembled grid to a compressed .npz for solver reuse."""
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            np.savez_compressed(filepath, **self.export_arrays())
            logger.info("Saved grid arrays to %s", filepath)
        except OSError as exc:
            raise GridBuilderError(f"Failed to save grid to '{filepath}': {exc}") from exc


# ---------------------------------------------------------------------------
# Example CoWoS-S stack builder (reference construction, not executed on import)
# ---------------------------------------------------------------------------

def build_cowos_s_reference_stack(gb: GridBuilder) -> None:
    """
    Reference helper wiring GridBuilder calls to the 6-layer CoWoS-S stack
    described in the TSMC packaging architecture reference:
      z=0            substrate/C4
      z=1..n_interposer   passive Si interposer (Cu routing homogenized as Si bulk)
      z=next block   microbump + underfill composite (anisotropic)
      z=next block   top chiplets (logic + HBM) -- Si bulk, power sources added separately
      z=next block   TIM
      z=top          heat sink / cold plate boundary (tagged, not power-bearing)
    Layer thicknesses must be pre-mapped to z-index ranges by the caller
    based on config/simulation_config.json layer definitions.
    """
    nz = gb.config.nz
    # Example fixed split; production use should read layer boundaries from config.
    z_substrate = slice(0, 1)
    z_interposer = slice(1, nz // 5)
    z_bump_underfill = slice(nz // 5, 2 * nz // 5)
    z_chiplet = slice(2 * nz // 5, 4 * nz // 5)
    z_tim = slice(4 * nz // 5, nz - 1)
    z_heatsink = slice(nz - 1, nz)

    gb.assign_layer(z_substrate, "copper")
    gb.assign_layer(z_interposer, "silicon")
    gb.assign_layer(z_bump_underfill, "bump_underfill_composite")
    gb.assign_layer(z_chiplet, "silicon")
    gb.assign_layer(z_tim, "tim")
    gb.assign_layer(z_heatsink, "copper")
    gb.tag_boundaries()


if __name__ == "__main__":
    # Minimal smoke test / usage demonstration.
    gb = GridBuilder(config={"grid": {"nx": 40, "ny": 40, "nz": 30,
                                       "dx": 25e-6, "dy": 25e-6, "dz": 10e-6}})
    build_cowos_s_reference_stack(gb)
    gb.add_power_source(power_watts=15.0,
                        x_slice=slice(10, 30), y_slice=slice(10, 30),
                        z_slice=slice(2 * gb.config.nz // 5, 4 * gb.config.nz // 5))
    gb.validate_grid()
    dt = gb.compute_cfl_dt(safety_factor=0.9)
    print(f"Stable explicit dt: {dt:.4e} s")