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
  - FAIL CLOSED. A run uses exactly the configuration it was given or it
    does not start. A missing or unreadable file, malformed JSON, and any
    unknown, retired, missing or mistyped key raise ConfigError naming the
    key. Nothing is ignored and nothing falls back to a default. This
    replaced a loader that fell back per section with a warning, which is
    how a retired key, a bottom-sink boundary and a 44 um mesh under a
    240 um stack all ran without an error.
  - Geometry is checked, never repaired: the authored stack must fill
    nz*dz exactly, in whole cells (StackSection.layer_cells).
  - A run has exactly one configuration source: a file (main.py --config),
    command-line flags, or the interactive prompts. Flags do not override a
    file. The dataclass defaults below serve programmatic construction on
    the flag and prompt paths; they never fill a gap in a file.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("RISC3D-AI.ConfigLoader")

DEFAULT_CONFIG_PATH = os.path.join("configs", "config.json")

# Relative tolerance for "these two lengths are the same", compared in SI.
# The micrometre/metre conversions carry ~1e-16 of round-off; anything above
# 1e-9 is a real geometric mismatch, not arithmetic.
GEOMETRY_REL_TOL = 1e-9


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

    layer_cells() is the single point where mesh resolution meets the
    stack, and it only ever CHECKS: the mesh must hold the authored stack
    exactly, in whole cells, or the run is refused. Choose dz so every
    layer is whole -- the defaults share a 20 um GCD, so any dz dividing
    20 um works (20, 10, 5, 2.5 um) -- and the geometry is then identical at
    every refinement level by construction.
    """
    interposer_um: float = 100.0
    bump_um: float = 40.0
    die_um: float = 100.0

    @property
    def total_um(self) -> float:
        return self.interposer_um + self.bump_um + self.die_um

    def layer_cells(self, nz: int, dz: float) -> Tuple[int, int, int]:
        """(interposer, composite, die) cell counts on a mesh of `nz` cells of
        `dz` METRES, or ConfigError if that mesh cannot hold the stack as
        authored. Both conditions are hard errors, compared in SI:

          - the stack fills the mesh: |nz*dz - total| / (nz*dz) < 1e-9
          - every layer is a whole number of cells, and at least one.

        There is no rounding and no clamping. Both used to happen here, and
        each silently models a different stack from the one authored: the
        default 240 um stack on a 120 um mesh was built as 100/10/10 um
        (composite -75%, die -90%), and the shipped configs/config.json --
        44 cells of 1 um -- as 42/1/1 um.
        """
        if nz < 3:
            raise ConfigError(
                f"the stack needs at least 3 z-cells (interposer / composite / die), "
                f"got nz={nz}")
        if not dz > 0:
            raise ConfigError(f"grid.dz must be > 0 m, got {dz!r}")
        mesh_m = nz * dz
        total_m = self.total_um * 1e-6
        if abs(mesh_m - total_m) / mesh_m >= GEOMETRY_REL_TOL:
            raise ConfigError(
                f"stack thickness {self.total_um:.3f} um ({self.interposer_um:g} interposer "
                f"+ {self.bump_um:g} composite + {self.die_um:g} die) does not equal the grid "
                f"z-extent nz*dz = {nz} x {dz * 1e6:.3f} um = {mesh_m * 1e6:.3f} um. The mesh "
                f"must hold the authored stack exactly: choose nz and dz with nz*dz = "
                f"{self.total_um:.3f} um.")
        cells = []
        for name, t_um in (("interposer_um", self.interposer_um),
                           ("bump_um", self.bump_um),
                           ("die_um", self.die_um)):
            t_m = t_um * 1e-6
            n = t_m / dz
            k = int(round(n))
            if not t_m > 0 or k < 1 or abs(t_m - k * dz) > GEOMETRY_REL_TOL * t_m:
                raise ConfigError(
                    f"stack.{name} = {t_um:.3f} um is {n:.4f} cells of dz = {dz * 1e6:.3f} um, "
                    f"not a whole number of cells (at least one). Rounding it would model a "
                    f"different layer; choose dz so every layer is whole.")
            cells.append(k)
        if sum(cells) != nz:     # unreachable when both checks pass; guarded anyway
            raise ConfigError(f"stack cells {cells} sum to {sum(cells)}, not nz = {nz}")
        return cells[0], cells[1], cells[2]

    def resolve_layer_boundaries(self, nz: int, dz: float) -> Tuple[int, int]:
        """(z_interposer_end, z_bump_end) indices on a mesh of `nz` cells of
        `dz` metres, such that

            interposer = z[0 : z_interposer_end]
            composite  = z[z_interposer_end : z_bump_end]
            die        = z[z_bump_end : nz]

        Raises ConfigError (from layer_cells) if the mesh cannot hold the
        stack exactly. This used to round each interface to the nearest cell
        face and clamp every layer to at least one cell, which built a
        different stack instead of refusing.
        """
        n_i, n_b, _ = self.layer_cells(nz, dz)
        return n_i, n_i + n_b

    def realized_thicknesses_um(self, nz: int, dz: float) -> Dict[str, float]:
        """{'interposer_um', 'bump_um', 'die_um', 'total_um'} as discretized on
        this mesh -- equal to the authored values (to GEOMETRY_REL_TOL),
        because layer_cells() refuses any mesh where they would differ.
        Reported next to the authored values so that is visible, not assumed."""
        n_i, n_b, n_d = self.layer_cells(nz, dz)
        dz_um = dz * 1e6
        return {
            "interposer_um": n_i * dz_um,
            "bump_um": n_b * dz_um,
            "die_um": n_d * dz_um,
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

        # The grid's z-extent and the authored stack are the same physical
        # object, so they must agree exactly, in whole cells -- a hard error.
        # This used to be a warning, after which the stack was rounded and
        # clamped to whatever the mesh could hold: a different stack,
        # reported only in a log line. A refinement study varies nz and dz
        # TOGETHER with nz*dz fixed, so it never needs a mismatch.
        s.layer_cells(self.grid.nz, self.grid.dz)

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

# Keys removed from the schema: section -> {retired key: (replacement, note)}.
# Git history (11ac3ff) shows these two are the only fields ever removed.
# A retired key is an error that names its replacement -- never dropped.
RETIRED_KEYS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "stack": {
        "interposer_frac": ("interposer_um", "a physical thickness in micrometres, not a "
                                             "fraction of nz"),
        "bump_frac": ("bump_um", "a physical thickness in micrometres, not a fraction of nz"),
    },
}

# The file's sections, in order, and the dataclass each one builds.
SECTIONS: Tuple[Tuple[str, type], ...] = (
    ("grid", GridSection),
    ("boundary_conditions", BoundaryConditionsSection),
    ("stack", StackSection),
    ("power", PowerSection),
    ("solver", SolverSection),
    ("visualization", VisualizationSection),
)


def _check_type(section: str, key: str, value: Any, default: Any) -> Any:
    """Checks one value against the type of its dataclass default and returns
    it (ints promoted to float where a float is expected). bool is rejected
    where a number is expected, since JSON true/false would otherwise pass
    as 1/0. A None default marks an Optional[float] field."""
    where = f"{section}.{key}"
    number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if default is None:
        if value is None:
            return None
        if not number:
            raise ConfigError(f"{where} must be a number or null, got {value!r}")
        return float(value)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false, got {value!r}")
        return value
    if isinstance(default, int):
        if not number or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
        return value
    if isinstance(default, float):
        if not number:
            raise ConfigError(f"{where} must be a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigError(f"{where} must be a string, got {value!r}")
        return value
    raise ConfigError(f"{where}: no type rule for a default of {default!r}")   # schema bug


def _build_section(dataclass_type, raw: Any, section_name: str):
    """Builds one section dataclass from its JSON object. Fail-closed: a
    retired, unknown, missing or mistyped key raises ConfigError naming it.
    Nothing is ignored, and no dataclass default fills a gap in the file."""
    if not isinstance(raw, dict):
        raise ConfigError(f"config section '{section_name}' must be a JSON object, "
                          f"got {type(raw).__name__}")
    schema = {f.name: f.default for f in fields(dataclass_type)}
    retired = RETIRED_KEYS.get(section_name, {})
    stale = sorted(k for k in raw if k in retired)
    if stale:
        raise ConfigError("; ".join(
            f"{section_name}.{k} is retired: use {section_name}.{retired[k][0]}, "
            f"{retired[k][1]}" for k in stale))
    unknown = sorted(k for k in raw if k not in schema)
    if unknown:
        raise ConfigError(
            f"unknown key(s) {', '.join(f'{section_name}.{k}' for k in unknown)}; "
            f"valid keys in '{section_name}' are {', '.join(sorted(schema))}")
    missing = [k for k in schema if k not in raw]
    if missing:
        raise ConfigError(
            f"missing required key(s) {', '.join(f'{section_name}.{k}' for k in missing)}; "
            f"a config file must state every value -- none is defaulted")
    return dataclass_type(**{k: _check_type(section_name, k, raw[k], schema[k])
                             for k in schema})


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """
    Loads and validates the full pipeline configuration from `path`, or from
    DEFAULT_CONFIG_PATH when none is given.

    Fail-closed: a missing or unreadable file, malformed JSON, a top level
    that is not an object, an unknown or missing section, and any unknown,
    retired, missing or mistyped key all raise ConfigError. There is no
    fallback to hardcoded defaults -- a run uses exactly this file, or it
    does not start. validate() then enforces the physical invariants.
    """
    resolved_path = path if path is not None else DEFAULT_CONFIG_PATH
    if not os.path.exists(resolved_path):
        raise ConfigError(f"config file '{resolved_path}' does not exist")
    try:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file '{resolved_path}' is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"config file '{resolved_path}' could not be read: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"config file '{resolved_path}' must hold a JSON object at top level, "
                          f"got {type(raw).__name__}")

    names = [name for name, _ in SECTIONS]
    unknown = sorted(set(raw) - set(names))
    if unknown:
        raise ConfigError(f"unknown config section(s) {', '.join(unknown)} in "
                          f"'{resolved_path}'; valid sections are {', '.join(names)}")
    missing = [name for name in names if name not in raw]
    if missing:
        raise ConfigError(f"missing required config section(s) {', '.join(missing)} in "
                          f"'{resolved_path}'")

    config = PipelineConfig(**{name: _build_section(cls, raw[name], name)
                               for name, cls in SECTIONS})
    config.validate()
    logger.info("Loaded pipeline configuration from %s", resolved_path)
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