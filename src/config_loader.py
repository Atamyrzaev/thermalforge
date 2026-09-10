"""
config_loader.py
RISC3D-AI :: Centralized Pipeline Configuration

Single source of truth for pipeline configuration. Loads configs/config.json
(or a caller-specified path) into a validated PipelineConfig, so GridBuilder's
grid/BC dict, ThermalSolver's SolverBCConfig, and the chiplet-stack/solver/
visualization parameters main.py needs are all derived from ONE consistent
schema -- instead of being redefined ad hoc as argparse defaults scattered
across main.py.

Design goals:
  - `python main.py` with zero CLI flags must run end-to-end using
    configs/config.json defaults.
  - Missing file / malformed JSON / missing keys must never crash the
    pipeline silently -- fall back to hardcoded safe defaults per-section
    with a clear warning, exactly like GridBuilder._load_config() already
    does for its own grid/BC section.
  - main.py's optional CLI flags override individual fields of the loaded
    config without needing to duplicate the whole schema.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.ConfigLoader")

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.json")


class ConfigError(Exception):
    """Raised on unrecoverable configuration errors (invalid types/ranges
    that cannot be safely defaulted)."""


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GridSection:
    """Voxel grid. The defaults describe a 20 x 20 mm advanced-packaging
    module with a 240 um stack -- the scale the Stage 2 chiplet catalog is
    authored for.

    These were previously 60x60x40 at 25/25/10 um, which is a 1.5 x 1.5 mm
    die. That is a perfectly good Stage 1 test case but only 2.25 mm^2, so
    Stage 2's 192 mm^2 catalog could not fit and `--optimize` aborted on
    every default run. Chiplet footprints are physical and do not shrink to
    fit a mesh (see optimizer.ChipletSpec), so the die had to grow instead.

    Chosen so that, simultaneously:
      - nx*dx = ny*dy = 20 mm, matching the catalog's package scale
      - nz*dz = 240 um = the authored 100/40/100 um stack exactly, with each
        layer a whole number of cells at dz=20 um (5/2/5)
      - nx*ny*nz = 19,200 <= thermal_solver.DIRECT_SOLVE_MAX_UNKNOWNS, so the
        steady-state solve takes the cached direct-LU path. Measured here:
        ~1.0-1.5 s for the first solve (which pays the factorization) and
        0.019-0.041 s for every solve after it, against ~4.7 s per solve on
        the conjugate-gradient path at 38,400 unknowns. Two orders of
        magnitude, and it is what makes an exhaustive placement search
        practical -- exhaustive precompute is 234 solves.
    """
    nx: int = 40
    ny: int = 40
    nz: int = 12
    dx: float = 500e-6
    dy: float = 500e-6
    dz: float = 20e-6


@dataclass
class BoundaryConditionsSection:
    t_ambient_c: float = 25.0
    h_convective: float = 2500.0
    # Sink on the TOP face by default: a CoWoS part cools through the die
    # backside into TIM1/lid/TIM2/heatsink, not down through the interposer.
    # Exactly one of bottom_bc / top_bc is the sink; the other is adiabatic.
    bottom_bc: str = "neumann"    # "robin" | "dirichlet" | "neumann"
    top_bc: str = "robin"         # "robin" | "dirichlet" | "neumann"
    side_bc: str = "neumann"      # only "neumann" currently supported


@dataclass
class StackSection:
    """Vertical chiplet stack layout, authored as PHYSICAL thicknesses in
    micrometers (bottom to top): thinned Si interposer, microbump+underfill
    composite, thinned Si die.

    Thicknesses are physical for the same reason chiplet footprints are (see
    src.optimizer.ChipletSpec): a layer is a real piece of material and its
    thickness cannot be a function of how finely the solver meshes the
    domain.

    The previous fraction-based authoring (interposer_frac / bump_frac,
    resolved via round(nz * frac)) made the realized bump thickness a
    rounding artifact of nz. Measured across nz, holding the nominal 0.15
    fraction fixed:

        nz    bump voxels    realized frac    vs nominal 0.15
        12         2            0.1667            +11.1%
        16         3            0.1875            +25.0%
        20         3            0.1500              0.0%
        24         3            0.1250            -16.7%
        28         4            0.1429             -4.8%
        32         5            0.1562             +4.2%
        40         6            0.1500              0.0%

    That is a -16.7%..+25.0% swing in the thickness of the single most
    thermally resistive layer in the stack -- the composite's kz is 13.1
    W/mK (derived from Rao et al. 2023, doi:10.2298/TSCI220805061R; see
    src/composite_materials.py) against silicon's 150, so a thickness error
    there moves the stack's resistance more than one anywhere else. A mesh
    refinement study run on fraction-authored layers therefore changes the
    GEOMETRY as it refines, and cannot separate discretization error from a
    silently different stack.

    resolve_layer_boundaries() converts these thicknesses to z-indices
    against a specific dz, which is the single point where mesh resolution
    is allowed to enter. Choose dz so each thickness is a whole number of
    cells (the defaults share a 20 um GCD, exact at dz = 20, 10, 5 and
    2.5 um) and the geometry is then identical at every refinement level.
    """
    interposer_um: float = 100.0
    bump_um: float = 40.0
    die_um: float = 100.0

    @property
    def total_um(self) -> float:
        return self.interposer_um + self.bump_um + self.die_um

    def resolve_layer_boundaries(self, nz: int, dz: float):
        """Resolves the stack to (z_interposer_end, z_bump_end) voxel indices
        on a mesh of spacing `dz` metres with `nz` layers.

        Returns indices such that:
            interposer = z[0 : z_interposer_end]
            composite  = z[z_interposer_end : z_bump_end]
            die        = z[z_bump_end : nz]

        Rounds each interface to the nearest cell face and guarantees every
        layer is at least one voxel thick, so a mesh too coarse to represent
        the stack still yields a valid three-region grid rather than an empty
        layer. realized_thicknesses_um() reports what was actually built.
        """
        if nz < 3:
            raise ConfigError(
                f"stack requires at least 3 z-layers to represent interposer / "
                f"composite / die, got nz={nz}"
            )
        dz_um = dz * 1e6
        z_i = int(round(self.interposer_um / dz_um))
        z_b = int(round((self.interposer_um + self.bump_um) / dz_um))

        # Clamp so all three layers are non-empty and in order.
        z_i = max(1, min(z_i, nz - 2))
        z_b = max(z_i + 1, min(z_b, nz - 1))
        return z_i, z_b

    def realized_thicknesses_um(self, nz: int, dz: float):
        """Returns {'interposer_um', 'bump_um', 'die_um', 'total_um'} as
        ACTUALLY discretized on this mesh, for reporting alongside any result
        so a geometry change cannot hide inside a refinement study."""
        z_i, z_b = self.resolve_layer_boundaries(nz, dz)
        dz_um = dz * 1e6
        return {
            "interposer_um": z_i * dz_um,
            "bump_um": (z_b - z_i) * dz_um,
            "die_um": (nz - z_b) * dz_um,
            "total_um": nz * dz_um,
        }


@dataclass
class PowerSection:
    die_power_w: float = 12.0
    die_footprint_frac: float = 0.5   # fraction of XY extent, centered


@dataclass
class SolverSection:
    method: str = "explicit"              # "explicit" | "crank_nicolson"
    steady_state: bool = False            # if True, solve directly for equilibrium (skips time-stepping)
    total_time_s: Optional[float] = None  # None -> auto = 400 * CFL dt (explicit) or total_time_s/200 (CN)
    dt: Optional[float] = None            # None -> auto-selected per method
    safety_factor: float = 0.9
    record_steps: int = 50
    max_steps: int = 2_000_000


@dataclass
class VisualizationSection:
    output_dir: str = "outputs"
    hotspot_threshold_c: float = 85.0
    use_pyvista: bool = True


@dataclass
class PipelineConfig:
    grid: GridSection
    boundary_conditions: BoundaryConditionsSection
    stack: StackSection
    power: PowerSection
    solver: SolverSection
    visualization: VisualizationSection

    # ------------------------------------------------------------------
    def validate(self) -> None:
        g = self.grid
        if min(g.nx, g.ny, g.nz) <= 0:
            raise ConfigError(f"grid.nx/ny/nz must be positive, got ({g.nx}, {g.ny}, {g.nz})")
        if min(g.dx, g.dy, g.dz) <= 0:
            raise ConfigError(f"grid.dx/dy/dz must be positive, got ({g.dx}, {g.dy}, {g.dz})")

        bc = self.boundary_conditions
        valid_bc = ("robin", "dirichlet", "neumann")
        for name, value in (("bottom_bc", bc.bottom_bc), ("top_bc", bc.top_bc)):
            if value not in valid_bc:
                raise ConfigError(f"boundary_conditions.{name} must be one of {valid_bc}, "
                                   f"got '{value}'")
        sink_faces = [n for n, v in (("bottom_bc", bc.bottom_bc), ("top_bc", bc.top_bc))
                      if v != "neumann"]
        if len(sink_faces) != 1:
            raise ConfigError(
                f"Exactly one of boundary_conditions.bottom_bc / top_bc must be the heat "
                f"sink ('robin' or 'dirichlet'), the other 'neumann'; got bottom_bc="
                f"'{bc.bottom_bc}', top_bc='{bc.top_bc}'.")
        if bc.side_bc != "neumann":
            raise ConfigError("boundary_conditions.side_bc: only 'neumann' is supported.")
        sink_kind = bc.bottom_bc if bc.bottom_bc != "neumann" else bc.top_bc
        if sink_kind == "robin" and bc.h_convective <= 0:
            raise ConfigError(f"boundary_conditions.h_convective must be > 0 for Robin BC, got {bc.h_convective}")

        s = self.stack
        for name, value in (("interposer_um", s.interposer_um),
                            ("bump_um", s.bump_um),
                            ("die_um", s.die_um)):
            if value <= 0:
                raise ConfigError(f"stack.{name} must be > 0 um, got {value}")

        # The grid's z-extent and the stack's authored thickness describe the
        # same physical object. A mismatch means the realized stack is not the
        # authored one, so warn loudly -- but do not raise: StackSection.
        # resolve_layer_boundaries() clamps to a valid three-region grid, and
        # main.py prints the realized thicknesses next to the authored ones,
        # so the discrepancy is visible rather than silent. Raising here would
        # make nz and dz un-varyable independently of the stack, which is
        # exactly what a mesh-refinement study needs to do.
        grid_total_um = self.grid.nz * self.grid.dz * 1e6
        if abs(grid_total_um - s.total_um) > 0.5 * self.grid.dz * 1e6:
            logger.warning(
                "Stack thickness (%.1f um = %.1f interposer + %.1f composite + %.1f die) "
                "does not match the grid z-extent (nz=%d x dz=%.2f um = %.1f um). The stack "
                "will be clamped to fit; realized thicknesses are reported at build time. "
                "Set nz*dz equal to the authored stack thickness to model it exactly.",
                s.total_um, s.interposer_um, s.bump_um, s.die_um,
                self.grid.nz, self.grid.dz * 1e6, grid_total_um,
            )

        p = self.power
        if p.die_power_w < 0:
            raise ConfigError(f"power.die_power_w must be >= 0, got {p.die_power_w}")
        if not (0.0 < p.die_footprint_frac <= 1.0):
            raise ConfigError(f"power.die_footprint_frac must be in (0, 1], got {p.die_footprint_frac}")

        sv = self.solver
        if sv.method not in ("explicit", "crank_nicolson"):
            raise ConfigError(f"solver.method must be 'explicit' or 'crank_nicolson', got '{sv.method}'")
        if sv.total_time_s is not None and sv.total_time_s <= 0:
            raise ConfigError(f"solver.total_time_s must be positive if set, got {sv.total_time_s}")
        if sv.dt is not None and sv.dt <= 0:
            raise ConfigError(f"solver.dt must be positive if set, got {sv.dt}")
        if not (0.0 < sv.safety_factor <= 1.0):
            raise ConfigError(f"solver.safety_factor must be in (0, 1], got {sv.safety_factor}")
        if sv.record_steps < 1:
            raise ConfigError(f"solver.record_steps must be >= 1, got {sv.record_steps}")
        if sv.max_steps < 1:
            raise ConfigError(f"solver.max_steps must be >= 1, got {sv.max_steps}")

        v = self.visualization
        if v.hotspot_threshold_c is None:
            raise ConfigError("visualization.hotspot_threshold_c must be set.")

    # ------------------------------------------------------------------
    def to_grid_builder_dict(self) -> Dict[str, Any]:
        """Converts to the nested dict schema GridBuilder(config=...) expects."""
        return {
            "grid": asdict(self.grid),
            "boundary_conditions": {
                "t_ambient_k": self.boundary_conditions.t_ambient_c + 273.15,
                "h_convective": self.boundary_conditions.h_convective,
                "bottom_bc": self.boundary_conditions.bottom_bc,
                "top_bc": self.boundary_conditions.top_bc,
                "side_bc": self.boundary_conditions.side_bc,
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Alias for to_grid_builder_dict(). GridBuilder.__init__() only accepts
        a raw dict, a JSON file path, or None -- it does NOT accept a
        PipelineConfig instance directly. Callers (including tests) that
        hold a PipelineConfig must construct GridBuilder via:

            gb = GridBuilder(config=pipeline_config.to_dict())

        rather than `GridBuilder(config=pipeline_config)`, which raises
        TypeError. This alias exists purely for that call-site clarity.
        """
        return self.to_grid_builder_dict()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "grid": asdict(self.grid),
            "boundary_conditions": asdict(self.boundary_conditions),
            "stack": asdict(self.stack),
            "power": asdict(self.power),
            "solver": asdict(self.solver),
            "visualization": asdict(self.visualization),
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _build_section(dataclass_type, raw: Dict[str, Any], section_name: str):
    """Builds a section dataclass from a raw dict, ignoring unknown keys
    (with a warning) and falling back to the dataclass default for any
    missing key -- never raises on partial/incomplete sections."""
    valid_keys = {f.name for f in fields(dataclass_type)}
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        logger.warning("Ignoring unknown key(s) in config section '%s': %s", section_name, sorted(unknown))
    filtered = {k: v for k, v in raw.items() if k in valid_keys}
    try:
        return dataclass_type(**filtered)
    except TypeError as exc:
        raise ConfigError(f"Invalid value type in config section '{section_name}': {exc}") from exc


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """
    Loads and validates the full pipeline configuration.

    Resolution order:
      1. `path` if explicitly given.
      2. DEFAULT_CONFIG_PATH ("configs/config.json") if it exists.
      3. Hardcoded dataclass defaults for every section, with a warning,
         if neither of the above is available or the file is malformed.

    This mirrors GridBuilder._load_config()'s fallback behavior so the
    pipeline never hard-crashes on a missing/partial config file.
    """
    resolved_path = path or DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}

    if path is not None and not os.path.exists(path):
        raise ConfigError(f"Explicitly specified config path does not exist: '{path}'")

    if os.path.exists(resolved_path):
        try:
            with open(resolved_path, "r") as f:
                raw = json.load(f)
            logger.info("Loaded pipeline configuration from %s", resolved_path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to parse config at '%s' (%s); falling back to hardcoded defaults.",
                         resolved_path, exc)
            raw = {}
    else:
        logger.warning(
            "Config file '%s' not found; running with hardcoded pipeline defaults. "
            "Create configs/config.json (see config_loader.write_default_config()) to customize.",
            resolved_path,
        )

    if not isinstance(raw, dict):
        logger.error("Top-level config content is not a JSON object; falling back to hardcoded defaults.")
        raw = {}

    config = PipelineConfig(
        grid=_build_section(GridSection, raw.get("grid", {}), "grid"),
        boundary_conditions=_build_section(
            BoundaryConditionsSection, raw.get("boundary_conditions", {}), "boundary_conditions"
        ),
        stack=_build_section(StackSection, raw.get("stack", {}), "stack"),
        power=_build_section(PowerSection, raw.get("power", {}), "power"),
        solver=_build_section(SolverSection, raw.get("solver", {}), "solver"),
        visualization=_build_section(VisualizationSection, raw.get("visualization", {}), "visualization"),
    )
    config.validate()
    return config


def write_default_config(path: str = DEFAULT_CONFIG_PATH) -> str:
    """Writes the hardcoded default PipelineConfig to `path` as pretty JSON.
    Useful for bootstrapping a fresh configs/config.json."""
    default_config = PipelineConfig(
        grid=GridSection(), boundary_conditions=BoundaryConditionsSection(),
        stack=StackSection(), power=PowerSection(),
        solver=SolverSection(), visualization=VisualizationSection(),
    )
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(default_config.as_dict(), f, indent=2)
        logger.info("Wrote default configuration to %s", path)
        return path
    except OSError as exc:
        raise ConfigError(f"Failed to write default config to '{path}': {exc}") from exc


if __name__ == "__main__":
    # Smoke test: write and reload the default config.
    written_path = write_default_config(os.path.join("configs_smoke_test", "config.json"))
    reloaded = load_config(written_path)
    print("Reloaded config:", reloaded.as_dict())