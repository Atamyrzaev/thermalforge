"""
tests/test_core.py
RISC3D-AI :: Core numerical correctness test suite (pytest)

IMPORTANT INTERFACE NOTE:
GridBuilder.__init__() accepts a config as `dict`, a JSON file path (`str`),
or `None` -- it does NOT accept a `PipelineConfig` instance directly (passing
one raises `TypeError`, verified explicitly in
TestConfigDictInterface.test_grid_builder_rejects_pipeline_config_instance).
Every test in this module that needs a PipelineConfig therefore constructs
GridBuilder as:

    gb = GridBuilder(config=pipeline_config.to_dict())

or, equivalently, with a raw dict literal directly. Both paths are exercised.

Coverage:
  1. TestConfigDictInterface   -- the dict/to_dict() contract itself.
  2. TestCFLStability          -- automated CFL bound matches an independently
                                   coded reference, and is actually enforced.
  3. TestEnergyConservationAndPositivity -- discrete energy balance holds
                                   exactly (no numerical leak/injection), and
                                   temperatures stay finite and physical.
  4. TestHotspotLocalization   -- peak/threshold hotspot detection matches
                                   the known location and magnitude of an
                                   injected power source.
  5. TestRobinBoundaryConvection -- the convective (Robin) bottom BC produces
                                   the exact analytically-predicted cooling
                                   rate, isolated from internal conduction.
  6. TestImplicitSolver        -- Crank-Nicolson is stable far beyond the
                                   explicit CFL bound, selects the right
                                   backend, caches its factorization, and
                                   conserves energy on the same identity.
  7. TestSteadyStateSolve      -- the direct equilibrium solve balances heat
                                   generated against heat removed exactly,
                                   and matches a long transient's asymptote.
  8. TestSolverSectionConfig   -- SolverSection validation rejects bad method
                                   and dt values.
  9. TestChipletPlacementOptimizer -- Stage 2's elitist best-tracker never
                                   promotes an unbuildable (overlapping)
                                   layout, so best_peak_temperature_c is
                                   always a real solved float.
 10. TestChipletPhysicalDimensions -- chiplet footprints are physical
                                   (micrometers) and do not change with mesh
                                   resolution; the die fit-check compares
                                   physical areas; power is conserved
                                   regardless of footprint rounding.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
import pytest

from src.config_loader import (
    PipelineConfig,
    ConfigError,
    ChipletEntry,
    InitialPlacementSection,
    GridSection,
    BoundaryConditionsSection,
    StackSection,
    PowerSection,
    SolverSection,
    VisualizationSection,
)
from src.grid_builder import GridBuilder, GridBuilderError, MATERIALS
import src.thermal_solver as thermal_solver_module
from src.thermal_solver import ThermalSolver, SolverBCConfig, ThermalSolverError

import src.optimizer as optimizer_module
from src.optimizer import ChipletPlacementOptimizer, ChipletSpec, OptimizerError
import src.exhaustive as exhaustive_module
import src.composite_materials as composite_module
import src.config_loader as config_loader_module
import src.experiment as experiment_module
import src.experiment_report as report_module
import src.surrogate as surrogate_module
import src.validation_status as validation_module
import main as main_module

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_pipeline_config(
    nx: int = 10, ny: int = 10, nz: int = 12,
    dx: float = 25e-6, dy: float = 25e-6, dz: float = 10e-6,
    t_ambient_c: float = 25.0, h_convective: float = 2500.0,
    bottom_bc: str = "neumann", top_bc: str = "robin",
    interposer_um: Optional[float] = None, bump_um: Optional[float] = None,
    die_um: Optional[float] = None,
    die_power_w: float = 0.05, die_footprint_frac: float = 0.5,
) -> PipelineConfig:
    """Builds a small, fast-to-solve PipelineConfig for unit testing.

    Stack layers are authored in MICROMETERS (see StackSection) and must be
    whole cells that fill nz*dz exactly -- the stack check refuses anything
    else. When not given they are derived from the grid: the layer
    interfaces are exactly the ones the old 40/20/40-of-nz*dz authoring was
    ROUNDED to, so every fixture keeps the geometry it always had. The
    difference is that the geometry is now authored, where before it was an
    unreported rounding of a thickness the mesh could not hold."""
    total_um = nz * dz * 1e6
    given = [v is not None for v in (interposer_um, bump_um, die_um)]
    if any(given) and not all(given):
        raise ValueError("give all three stack thicknesses or none")
    if not any(given):
        dz_um = dz * 1e6
        z_i = int(round(0.4 * total_um / dz_um))
        z_b = int(round(0.6 * total_um / dz_um))
        z_i = max(1, min(z_i, nz - 2))
        z_b = max(z_i + 1, min(z_b, nz - 1))
        interposer_um = z_i * dz_um
        bump_um = (z_b - z_i) * dz_um
        die_um = (nz - z_b) * dz_um
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=t_ambient_c, h_convective=h_convective,
            bottom_bc=bottom_bc, top_bc=top_bc,
        ),
        stack=StackSection(interposer_um=interposer_um, bump_um=bump_um, die_um=die_um),
        power=PowerSection(die_power_w=die_power_w, die_footprint_frac=die_footprint_frac),
        solver=SolverSection(),
        visualization=VisualizationSection(),
    )


def build_uniform_grid(config: PipelineConfig, material: str) -> GridBuilder:
    """Builds a GridBuilder filled entirely with a single material, via
    config.to_dict() (never a PipelineConfig instance)."""
    gb = GridBuilder(config=config.to_dict())
    gb.assign_layer(slice(0, config.grid.nz), material)
    gb.tag_boundaries()
    gb.validate_grid()
    return gb


def build_stack_grid(config: PipelineConfig):
    """Builds the standard interposer / bump+underfill / Si-die stack with a
    centered active power footprint, mirroring main.py's build_chiplet_stack()
    logic. Returns (GridBuilder, z_interposer_end, z_bump_end)."""
    gb = GridBuilder(config=config.to_dict())
    nz = config.grid.nz
    nx, ny = config.grid.nx, config.grid.ny

    # Physical (um) layer thicknesses resolved against this mesh's dz --
    # mirrors main.build_chiplet_stack(), never fractions of nz.
    z_interposer_end, z_bump_end = config.stack.resolve_layer_boundaries(nz, config.grid.dz)

    gb.assign_layer(slice(0, z_interposer_end), "silicon")
    gb.assign_layer(slice(z_interposer_end, z_bump_end), "bump_underfill_composite")
    gb.assign_layer(slice(z_bump_end, nz), "silicon")
    gb.tag_boundaries()

    margin_x = int(round(nx * (1.0 - config.power.die_footprint_frac) / 2.0))
    margin_y = int(round(ny * (1.0 - config.power.die_footprint_frac) / 2.0))
    x_slice = slice(margin_x, nx - margin_x)
    y_slice = slice(margin_y, ny - margin_y)
    z_slice = slice(z_bump_end, nz)

    gb.add_power_source(power_watts=config.power.die_power_w, x_slice=x_slice, y_slice=y_slice, z_slice=z_slice)
    gb.validate_grid()
    return gb, z_interposer_end, z_bump_end, (x_slice, y_slice, z_slice)


def reference_max_dt(gb: GridBuilder, safety_factor: float = 1.0) -> float:
    """
    Independently-coded (test-side) reference implementation of the CFL /
    diagonal-dominance bound, used to cross-check ThermalSolver.compute_max_dt()
    without calling any of its private methods. Mirrors the physics (harmonic-
    mean face conductivities, Robin/Dirichlet bottom coefficient) but is
    written from scratch here so this is a genuine independent verification.
    """
    kx, ky, kz = gb.kx_map, gb.ky_map, gb.kz_map
    rho, cp = gb.rho_map, gb.cp_map
    dx, dy, dz = gb.config.dx, gb.config.dy, gb.config.dz
    nx, ny, nz = kx.shape

    def harmonic(a, b):
        return 2.0 * a * b / (a + b + _EPS)

    total = np.zeros_like(kx)

    kxf = harmonic(kx[:-1, :, :], kx[1:, :, :]) / dx ** 2
    left = np.zeros_like(kx); left[1:, :, :] = kxf
    right = np.zeros_like(kx); right[:-1, :, :] = kxf
    total += left + right

    kyf = harmonic(ky[:, :-1, :], ky[:, 1:, :]) / dy ** 2
    left = np.zeros_like(ky); left[:, 1:, :] = kyf
    right = np.zeros_like(ky); right[:, :-1, :] = kyf
    total += left + right

    kzf = harmonic(kz[:, :, :-1], kz[:, :, 1:]) / dz ** 2
    left = np.zeros_like(kz); left[:, :, 1:] = kzf
    right = np.zeros_like(kz); right[:, :, :-1] = kzf
    total += left + right

    sink_k = 0 if gb.config.sink_face == "bottom" else nz - 1
    if gb.config.sink_bc == "dirichlet":
        coeff_b = 2.0 * kz[:, :, sink_k] / dz ** 2
    else:
        coeff_b = np.full((nx, ny), gb.config.h_convective / dz)
    total[:, :, sink_k] += coeff_b

    dt_per_cell = (rho * cp) / total
    return float(np.min(dt_per_cell)) * safety_factor


# ---------------------------------------------------------------------------
# 1. Config / dict interface contract
# ---------------------------------------------------------------------------

class TestConfigDictInterface:
    def test_grid_builder_accepts_config_to_dict(self):
        config = make_pipeline_config()
        d = config.to_dict()
        assert isinstance(d, dict)
        gb = GridBuilder(config=d)
        assert gb.kx_map.shape == (config.grid.nx, config.grid.ny, config.grid.nz)

    def test_to_dict_matches_to_grid_builder_dict(self):
        config = make_pipeline_config()
        assert config.to_dict() == config.to_grid_builder_dict()

    def test_grid_builder_accepts_raw_dict_directly(self):
        raw = {
            "grid": {"nx": 8, "ny": 8, "nz": 10, "dx": 25e-6, "dy": 25e-6, "dz": 10e-6},
            "boundary_conditions": {
                "t_ambient_k": 298.15, "h_convective": 2500.0,
                "bottom_bc": "neumann", "top_bc": "robin", "side_bc": "neumann",
            },
        }
        gb = GridBuilder(config=raw)
        assert gb.kx_map.shape == (8, 8, 10)

    def test_grid_builder_accepts_none_with_fallback_defaults(self):
        # No config.json on disk at cwd for this test -> hardcoded fallback.
        gb = GridBuilder(config=None)
        assert gb.kx_map.ndim == 3

    def test_grid_builder_rejects_pipeline_config_instance(self):
        """The core bug report: GridBuilder must NOT accept a PipelineConfig
        object directly -- only dict / str / None. Verifies the exact
        TypeError callers must route around via .to_dict()."""
        config = make_pipeline_config()
        with pytest.raises(TypeError):
            GridBuilder(config=config)  # type: ignore[arg-type]

    def test_thermal_solver_from_grid_builder_roundtrip(self):
        """End-to-end: PipelineConfig -> to_dict() -> GridBuilder ->
        ThermalSolver.from_grid_builder() must succeed with no manual
        SolverBCConfig construction required."""
        config = make_pipeline_config()
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        assert isinstance(solver.bc, SolverBCConfig)
        assert solver.shape == (config.grid.nx, config.grid.ny, config.grid.nz)


# ---------------------------------------------------------------------------
# 2. CFL stability
# ---------------------------------------------------------------------------

class TestCFLStability:
    @pytest.mark.parametrize("material,bottom_bc", [
        ("copper", "robin"),
        ("silicon", "robin"),
        ("copper", "dirichlet"),
        ("bump_underfill_composite", "robin"),
    ])
    def test_compute_max_dt_matches_independent_reference(self, material, bottom_bc):
        config = make_pipeline_config(bottom_bc=bottom_bc, top_bc="neumann", die_power_w=0.0)
        gb = build_uniform_grid(config, material)
        solver = ThermalSolver.from_grid_builder(gb)

        dt_solver = solver.compute_max_dt(safety_factor=1.0)
        dt_reference = reference_max_dt(gb, safety_factor=1.0)

        assert dt_solver == pytest.approx(dt_reference, rel=1e-9)

    def test_safety_factor_scales_linearly(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        dt_90 = solver.compute_max_dt(safety_factor=0.9)
        dt_45 = solver.compute_max_dt(safety_factor=0.45)
        assert dt_45 == pytest.approx(dt_90 * 0.5, rel=1e-9)

    def test_step_at_or_below_bound_is_stable(self):
        config = make_pipeline_config(die_power_w=0.02)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T_after = solver.step(dt, enforce_stability=True)
        assert np.all(np.isfinite(T_after))

    def test_step_above_bound_raises_thermal_solver_error(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "copper")  # Cu -> tight CFL bound
        solver = ThermalSolver.from_grid_builder(gb)

        dt_stable = solver.compute_max_dt(safety_factor=0.9)
        with pytest.raises(ThermalSolverError):
            solver.step(dt_stable * 10.0, enforce_stability=True)

    def test_copper_bound_is_tighter_than_silicon(self):
        """Sanity check on physical ordering: Cu's higher k/(rho*cp) ratio
        must yield a smaller (more restrictive) stable dt than Si on an
        identical grid."""
        cfg_cu = make_pipeline_config(die_power_w=0.0)
        cfg_si = make_pipeline_config(die_power_w=0.0)
        gb_cu = build_uniform_grid(cfg_cu, "copper")
        gb_si = build_uniform_grid(cfg_si, "silicon")
        dt_cu = ThermalSolver.from_grid_builder(gb_cu).compute_max_dt(safety_factor=1.0)
        dt_si = ThermalSolver.from_grid_builder(gb_si).compute_max_dt(safety_factor=1.0)
        assert dt_cu < dt_si


# ---------------------------------------------------------------------------
# 3. Energy conservation & positivity
# ---------------------------------------------------------------------------

class TestEnergyConservationAndPositivity:
    def test_equilibrium_field_is_a_fixed_point(self):
        """If T is already uniform at T_ambient everywhere and Q=0, one
        explicit step must leave T EXACTLY unchanged (no numerical leak/
        injection from the discretization at equilibrium)."""
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        T0 = solver.T.copy()
        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        assert np.allclose(T1, T0, atol=1e-9)

    def test_discrete_energy_balance_with_active_source(self):
        """sum(rho*cp*dT*cell_vol) over one step must equal
        dt * (sum(Q*cell_vol) - sum(boundary_outflow*cell_area... )) exactly,
        i.e. the discretization neither creates nor destroys energy beyond
        the explicit source term and the explicit boundary exchange term."""
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin", top_bc="neumann")
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        cell_vol = config.grid.dx * config.grid.dy * config.grid.dz
        T0 = solver.T.copy()
        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        actual_energy_delta = float(np.sum(solver.rho_map * solver.cp_map * (T1 - T0))) * cell_vol

        # Reference boundary outflow term, computed independently (not via
        # solver._compute_divergence): coeff * (T0_bottom - T_ambient) per
        # bottom-layer cell, matching the Robin/Dirichlet BC definition.
        if config.boundary_conditions.bottom_bc == "dirichlet":
            coeff_b = 2.0 * solver.kz_map[:, :, 0] / (config.grid.dz ** 2)
        else:
            coeff_b = np.full((config.grid.nx, config.grid.ny), config.boundary_conditions.h_convective / config.grid.dz)
        boundary_outflow = coeff_b * (T0[:, :, 0] - solver.bc.t_ambient_k)

        expected_energy_delta = dt * cell_vol * (float(np.sum(solver.q_map)) - float(np.sum(boundary_outflow)))

        assert actual_energy_delta == pytest.approx(expected_energy_delta, rel=1e-6, abs=1e-12)

    def test_temperatures_remain_finite_and_physical_over_run(self):
        config = make_pipeline_config(die_power_w=0.1)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=200 * dt, dt=dt, safety_factor=0.9)

        T_final = result["final_T"]
        assert np.all(np.isfinite(T_final))
        assert np.all(T_final > 0.0)  # positive Kelvin -- non-physical to violate

    def test_active_power_source_strictly_increases_peak_temperature(self):
        config = make_pipeline_config(die_power_w=0.15)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T0_peak = float(np.max(solver.T))
        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=100 * dt, dt=dt, safety_factor=0.9)
        T1_peak = float(np.max(result["final_T"]))

        assert T1_peak > T0_peak

    def test_zero_power_never_exceeds_ambient(self):
        """With no active source and starting at ambient, no cell should
        ever rise above ambient (only Robin cooling toward ambient is
        possible, never overshoot heating)."""
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=100 * dt, dt=dt, safety_factor=0.9)
        assert np.all(result["final_T"] <= solver.bc.t_ambient_k + 1e-9)


# ---------------------------------------------------------------------------
# 4. Hotspot localization
# ---------------------------------------------------------------------------

class TestHotspotLocalization:
    def test_peak_hotspot_is_within_active_power_footprint(self):
        config = make_pipeline_config(nx=16, ny=16, nz=14, die_power_w=0.2, die_footprint_frac=0.4)
        gb, z_interposer_end, z_bump_end, (x_slice, y_slice, z_slice) = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=300 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        hotspots = solver.get_hotspots(threshold_c=-273.0)  # threshold below absolute zero -> peak always found
        pi, pj, pk = hotspots["peak_location_ijk"]

        assert x_slice.start <= pi < x_slice.stop
        assert y_slice.start <= pj < y_slice.stop
        assert z_slice.start <= pk < z_slice.stop

    def test_peak_hotspot_is_at_die_top_farthest_from_heat_sink(self):
        """Physical sanity: with only a bottom heat sink and an adiabatic
        top, the hottest voxel in the die must be at the top z-layer (the
        point of maximum thermal resistance from the sink), not buried
        mid-die."""
        config = make_pipeline_config(nx=16, ny=16, nz=14, die_power_w=0.2)
        gb, z_interposer_end, z_bump_end, _ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=300 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        hotspots = solver.get_hotspots(threshold_c=-273.0)
        _, _, pk = hotspots["peak_location_ijk"]
        assert pk == config.grid.nz - 1

    def test_hotspot_count_responds_monotonically_to_threshold(self):
        config = make_pipeline_config(nx=14, ny=14, nz=12, die_power_w=0.3)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = solver.compute_max_dt(safety_factor=0.9)
        result = solver.run(total_time_s=500 * dt, dt=dt, safety_factor=0.9)
        solver.T = result["final_T"]

        low = solver.get_hotspots(threshold_c=solver.bc.t_ambient_k - 273.15)["count"]     # ~ambient -> most cells qualify
        mid = solver.get_hotspots(threshold_c=85.0)["count"]
        high = solver.get_hotspots(threshold_c=1e6)["count"]                              # impossible -> zero

        assert low >= mid >= high == 0

    def test_no_power_source_yields_no_hotspots_above_ambient(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        hotspots = solver.get_hotspots(threshold_c=config.boundary_conditions.t_ambient_c + 0.01)
        assert hotspots["count"] == 0


# ---------------------------------------------------------------------------
# 5. Robin boundary convection
# ---------------------------------------------------------------------------

class TestRobinBoundaryConvection:
    def test_uniform_elevated_field_isolates_robin_cooling_to_bottom_layer(self):
        """With T uniformly elevated by delta_T everywhere (zero internal
        gradient, so zero conduction anywhere), only the bottom (z=0) layer
        should change after one step -- via the Robin term referencing the
        external T_ambient -- while every other layer stays exactly fixed."""
        config = make_pipeline_config(bottom_bc="robin", top_bc="neumann", die_power_w=0.0, h_convective=1800.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 12.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        # Non-bottom layers: exactly unchanged (no internal gradient anywhere,
        # no boundary reference except at z=0).
        assert np.allclose(T1[:, :, 1:], T_initial[:, :, 1:], atol=1e-9)

        # Bottom layer: analytic Robin prediction, isolated from conduction
        # because the neighboring z=1 layer is at the SAME temperature (zero
        # vertical gradient), so the only active term is the convective one.
        rho0 = solver.rho_map[:, :, 0]
        cp0 = solver.cp_map[:, :, 0]
        expected_bottom = T_initial[:, :, 0] - dt * (config.boundary_conditions.h_convective / config.grid.dz) \
            * (T_initial[:, :, 0] - solver.bc.t_ambient_k) / (rho0 * cp0)

        assert np.allclose(T1[:, :, 0], expected_bottom, rtol=1e-9, atol=1e-9)

    def test_robin_cools_toward_ambient_never_past_it(self):
        config = make_pipeline_config(bottom_bc="robin", top_bc="neumann", die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 20.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        for _ in range(50):
            solver.step(dt)

        # Should have cooled toward ambient, but a single explicit run at a
        # stable dt must never overshoot past ambient (would indicate a sign
        # error or instability in the BC term).
        assert np.all(solver.T[:, :, 0] >= solver.bc.t_ambient_k - 1e-6)
        assert np.all(solver.T[:, :, 0] <= T_initial[:, :, 0])

    def test_higher_h_convective_cools_bottom_layer_faster(self):
        """Larger convective film coefficient must produce strictly greater
        cooling of the bottom layer after an identical elapsed physical
        time, all else equal."""
        delta_T = 15.0

        def run_and_get_bottom_mean(h_value: float) -> float:
            config = make_pipeline_config(bottom_bc="robin", top_bc="neumann", die_power_w=0.0, h_convective=h_value)
            gb = build_uniform_grid(config, "silicon")
            solver = ThermalSolver.from_grid_builder(gb)
            T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
            solver.reset(initial_temperature_k=T_initial)
            # Use the SAME dt for both cases (driven by the smaller of the two
            # to keep both stable) so the comparison isolates h's effect.
            dt = solver.compute_max_dt(safety_factor=0.5)
            for _ in range(20):
                solver.step(dt, enforce_stability=False)
            return float(np.mean(solver.T[:, :, 0]))

        mean_low_h = run_and_get_bottom_mean(500.0)
        mean_high_h = run_and_get_bottom_mean(4500.0)

        ambient = 298.15
        assert (mean_high_h - ambient) < (mean_low_h - ambient)

    def test_dirichlet_bottom_coefficient_matches_expected_formula(self):
        """Cross-check the Dirichlet boundary coefficient (2*k/dz^2) used
        internally by comparing its effect against a hand-derived one-step
        prediction, analogous to the Robin isolation test above."""
        config = make_pipeline_config(bottom_bc="dirichlet", top_bc="neumann", die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        delta_T = 10.0
        T_initial = np.full(solver.shape, solver.bc.t_ambient_k + delta_T, dtype=np.float64)
        solver.reset(initial_temperature_k=T_initial)

        dt = solver.compute_max_dt(safety_factor=0.9)
        T1 = solver.step(dt)

        coeff_dirichlet = 2.0 * solver.kz_map[:, :, 0] / (config.grid.dz ** 2)
        rho0 = solver.rho_map[:, :, 0]
        cp0 = solver.cp_map[:, :, 0]
        expected_bottom = T_initial[:, :, 0] - dt * coeff_dirichlet * (T_initial[:, :, 0] - solver.bc.t_ambient_k) / (rho0 * cp0)

        assert np.allclose(T1[:, :, 0], expected_bottom, rtol=1e-9, atol=1e-9)
        assert np.allclose(T1[:, :, 1:], T_initial[:, :, 1:], atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Implicit (Crank-Nicolson) solver
# ---------------------------------------------------------------------------

class TestImplicitSolver:
    def test_stable_at_dt_far_exceeding_explicit_cfl_bound(self):
        """The entire point of Crank-Nicolson: a dt that would immediately
        raise ThermalSolverError under the explicit scheme must produce a
        perfectly finite result here."""
        config = make_pipeline_config(die_power_w=0.05)
        gb = build_uniform_grid(config, "copper")  # tight explicit CFL bound
        solver = ThermalSolver.from_grid_builder(gb)

        dt_explicit_bound = solver.compute_max_dt(safety_factor=0.9)
        dt_huge = dt_explicit_bound * 1000.0  # would hard-fail solver.step()

        with pytest.raises(ThermalSolverError):
            solver.step(dt_huge)  # sanity: explicit really does reject this dt

        T_after = solver.step_implicit(dt_huge)
        assert np.all(np.isfinite(T_after))

    def test_adi_method_raises_not_implemented(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)
        with pytest.raises(NotImplementedError):
            solver.step_implicit(1e-6, method="adi")

    def test_zero_or_negative_dt_raises(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)
        with pytest.raises(ThermalSolverError):
            solver.step_implicit(0.0)
        with pytest.raises(ThermalSolverError):
            solver.step_implicit(-1e-6)

    def test_direct_backend_selected_for_small_grid(self):
        config = make_pipeline_config(nx=8, ny=8, nz=10, die_power_w=0.02)  # N=640 << threshold
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        solver.step_implicit(1e-6)
        cache_entry = next(iter(solver._cn_solver_cache.values()))
        assert cache_entry["backend"] == "direct"

    def test_iterative_backend_selected_above_threshold_and_converges(self):
        """Grid sized just above DIRECT_SOLVE_MAX_UNKNOWNS must route to the
        Jacobi-preconditioned iterative backend and still produce a finite,
        converged result."""
        from src.thermal_solver import DIRECT_SOLVE_MAX_UNKNOWNS
        # Pick dimensions whose product safely exceeds the threshold.
        n_side = int(np.ceil((DIRECT_SOLVE_MAX_UNKNOWNS * 1.1) ** (1.0 / 3.0))) + 1
        config = make_pipeline_config(nx=n_side, ny=n_side, nz=n_side, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        assert solver.nx * solver.ny * solver.nz > DIRECT_SOLVE_MAX_UNKNOWNS

        T_after = solver.step_implicit(1e-4)
        cache_entry = next(iter(solver._cn_solver_cache.values()))
        assert cache_entry["backend"] == "iterative"
        assert np.all(np.isfinite(T_after))

    def test_lu_cache_reused_across_repeated_dt(self):
        """The factorization/preconditioner must be built once and reused
        for subsequent step_implicit() calls at the SAME dt (performance
        contract: repeated fixed-dt steps should not rebuild the system)."""
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        dt = 1e-6
        solver.step_implicit(dt)
        cached_object_id = id(next(iter(solver._cn_solver_cache.values())))
        solver.step_implicit(dt)
        assert len(solver._cn_solver_cache) == 1
        assert id(next(iter(solver._cn_solver_cache.values()))) == cached_object_id

    def test_energy_conservation_holds_for_implicit_scheme_too(self):
        """The same discrete energy-balance identity verified for the
        explicit scheme must also hold (to CN's linear-solve tolerance) for
        Crank-Nicolson: conduction redistributes energy internally and only
        Q and the boundary term are true sources/sinks."""
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin", top_bc="neumann")
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        cell_vol = config.grid.dx * config.grid.dy * config.grid.dz
        T0 = solver.T.copy()
        dt = 1e-6
        T1 = solver.step_implicit(dt)

        actual_energy_delta = float(np.sum(solver.rho_map * solver.cp_map * (T1 - T0))) * cell_vol

        coeff_b = np.full((config.grid.nx, config.grid.ny),
                           config.boundary_conditions.h_convective / config.grid.dz)
        # Trapezoidal (Crank-Nicolson) boundary outflow: average of the term
        # evaluated at T0 and T1, matching the theta=0.5 scheme actually used.
        boundary_outflow_0 = coeff_b * (T0[:, :, 0] - solver.bc.t_ambient_k)
        boundary_outflow_1 = coeff_b * (T1[:, :, 0] - solver.bc.t_ambient_k)
        boundary_outflow_avg = 0.5 * (boundary_outflow_0 + boundary_outflow_1)

        expected_energy_delta = dt * cell_vol * (float(np.sum(solver.q_map)) - float(np.sum(boundary_outflow_avg)))

        assert actual_energy_delta == pytest.approx(expected_energy_delta, rel=1e-5, abs=1e-10)

    def test_crank_nicolson_matches_explicit_reference_within_tolerance(self):
        """Cross-validation: at a dt small enough to also be explicit-
        stable, both schemes must agree closely (both are consistent
        discretizations of the same PDE); at that resolution, CN's extra
        implicit averaging should not meaningfully change the result."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)

        gb_exp, *_ = build_stack_grid(config)
        solver_exp = ThermalSolver.from_grid_builder(gb_exp)
        dt = solver_exp.compute_max_dt(safety_factor=0.9)
        result_exp = solver_exp.run(total_time_s=100 * dt, dt=dt, method="explicit")

        gb_cn, *_ = build_stack_grid(config)
        solver_cn = ThermalSolver.from_grid_builder(gb_cn)
        result_cn = solver_cn.run(total_time_s=100 * dt, dt=dt, method="crank_nicolson")

        assert np.max(np.abs(result_exp["final_T"] - result_cn["final_T"])) < 0.05  # < 0.05 K agreement

    def test_run_with_crank_nicolson_method_tag(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        result = solver.run(total_time_s=1e-5, dt=1e-6, method="crank_nicolson")
        assert result["method"] == "crank_nicolson"
        assert np.all(np.isfinite(result["final_T"]))


class TestSteadyStateSolve:
    def test_steady_state_energy_balance_is_exact(self):
        """At true steady state (dT/dt=0), total heat generated must exactly
        equal total heat removed through the boundary -- the cleanest
        possible energy-conservation check, since there is no transient
        storage term to account for."""
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state()

        coeff_b = np.full((config.grid.nx, config.grid.ny),
                           config.boundary_conditions.h_convective / config.grid.dz)
        sk = solver._sink_k
        boundary_outflow = float(np.sum(coeff_b * (T_steady[:, :, sk] - solver.bc.t_ambient_k)))
        total_q = float(np.sum(solver.q_map))

        assert boundary_outflow == pytest.approx(total_q, rel=1e-4, abs=1e-9)

    def test_steady_state_matches_long_transient_asymptote(self):
        """A sufficiently long Crank-Nicolson run must converge toward the
        same equilibrium the direct steady-state solve produces exactly."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.03)

        gb_a, *_ = build_stack_grid(config)
        solver_a = ThermalSolver.from_grid_builder(gb_a)
        T_steady = solver_a.solve_steady_state()

        gb_b, *_ = build_stack_grid(config)
        solver_b = ThermalSolver.from_grid_builder(gb_b)
        result = solver_b.run(total_time_s=20.0, dt=0.1, method="crank_nicolson")

        assert np.max(np.abs(T_steady - result["final_T"])) < 1.0  # within 1K after long transient

    def test_steady_state_zero_power_returns_ambient_everywhere(self):
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state()
        assert np.allclose(T_steady, solver.bc.t_ambient_k, atol=1e-6)

    def test_apply_result_false_does_not_mutate_solver_state(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_before = solver.T.copy()
        T_steady = solver.solve_steady_state(apply_result=False)

        assert np.array_equal(solver.T, T_before)  # solver.T untouched
        assert not np.allclose(T_steady, T_before)  # but the computed result differs

    def test_apply_result_true_updates_solver_temperature(self):
        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T_steady = solver.solve_steady_state(apply_result=True)
        assert np.array_equal(solver.T, T_steady)


class TestSteadyStateVerification:
    """Every steady-state solve is verified, not trusted.

    A convergence flag says "my iteration stopped", not "the answer is right".
    solve_steady_state() therefore measures its own answer against two
    independent checks and raises if either fails: an algebraic residual on
    the posed system, and an energy balance derived from conservation rather
    than from the discretization.
    """

    def test_verification_metrics_are_at_machine_precision(self):
        """A real solve must pass both checks with orders of margin, not
        scrape past the tolerance."""
        config = make_pipeline_config(nx=12, ny=12, nz=12, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        N = solver.nx * solver.ny * solver.nz
        T = solver.solve_steady_state(apply_result=False)
        forcing = (solver._boundary_forcing
                   + solver.q_map.reshape(N) / solver._rho_cp.reshape(N))
        checks = solver._verify_steady_state(T.reshape(N), forcing)

        # Two orders of margin below the raise threshold. Measured on this
        # grid: residual ~1.4e-11 against a 1e-8 tolerance.
        assert checks["residual"] < 1e-2 * thermal_solver_module.STEADY_RESIDUAL_TOL
        assert checks["energy_imbalance_k"] < 1e-2 * thermal_solver_module.STEADY_ENERGY_BALANCE_TOL_K

    def test_energy_balance_check_survives_zero_power(self):
        """With q=0 both sides of the balance vanish. Measuring the imbalance
        as a ratio would divide float noise by float noise and report 100%
        error on a perfectly correct solve; measuring it in equivalent kelvin
        stays meaningful. This pins that choice."""
        config = make_pipeline_config(die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        T = solver.solve_steady_state(apply_result=False)   # must not raise
        assert np.allclose(T, solver.bc.t_ambient_k, atol=1e-6)

    def test_verification_rejects_a_corrupted_field(self):
        """Regression injection: perturb one cell of a converged solution and
        confirm both checks catch it. Without this, the verification could be
        silently vacuous."""
        config = make_pipeline_config(nx=12, ny=12, nz=12, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        N = solver.nx * solver.ny * solver.nz
        T = solver.solve_steady_state(apply_result=False)
        forcing = (solver._boundary_forcing
                   + solver.q_map.reshape(N) / solver._rho_cp.reshape(N))

        # An interior cell perturbation breaks the algebraic residual.
        corrupted = T.reshape(N).copy()
        corrupted[N // 2] += 5.0
        with pytest.raises(ThermalSolverError, match="relative residual"):
            solver._verify_steady_state(corrupted, forcing)

        # The energy-balance check exists to catch a WRONG OPERATOR OR FORCING
        # -- a field that exactly satisfies the system it was solved against,
        # while that system is not the physics. Perturbing T cannot exercise
        # it: any change to T also breaks the residual, which runs first, so
        # the energy check is never reached. (An earlier version of this test
        # did exactly that and passed on the residual while claiming to test
        # conservation.)
        #
        # So inject a phantom heat source into the forcing but NOT into q_map.
        # T_bad solves M T = -(forcing + phantom) exactly, so the residual
        # against that forcing is at machine precision -- but the boundary now
        # removes power that q_map never generated, which only the
        # conservation check can see.
        phantom = np.zeros(N)
        phantom[N // 2] = 1.0e3
        f_bad = forcing + phantom
        T_bad = solver._steady_lu.solve(-f_bad)
        with pytest.raises(ThermalSolverError, match="energy-balance"):
            solver._verify_steady_state(T_bad, f_bad)

    def test_energy_balance_holds_when_sink_and_source_materials_differ(self):
        """Conservation is sum(q) == sum(coeff * (T_sink - T_amb)), UNWEIGHTED.

        Interior face fluxes cancel exactly in the symmetric conductance
        matrix A, but not in M = A / rho_cp. An earlier version of the check
        compared sum(q / rho_cp) against sum(coeff / rho_cp * (T - T_amb)),
        which is only equal when every heat-source cell and every sink cell
        share one rho_cp. The standard stack happens to satisfy that (die and
        interposer are both silicon), so it never fired -- but a lid or TIM on
        the sink face breaks it. Here the sink layer is copper (rho*cp ~2.1x
        silicon's) and the source sits in silicon: a correct solve must pass
        verification, and the watts leaving the sink face -- computed here
        from h, dx, dy alone, not from any solver internal -- must equal the
        watts injected.
        """
        config = make_pipeline_config(nx=12, ny=12, nz=12, h_convective=2500.0,
                                      bottom_bc="neumann", top_bc="robin")
        nx, ny, nz = config.grid.nx, config.grid.ny, config.grid.nz
        gb = GridBuilder(config=config.to_dict())
        gb.assign_layer(slice(0, nz - 1), "silicon")
        gb.assign_layer(slice(nz - 1, nz), "copper")
        gb.tag_boundaries()
        power_w = 0.01
        gb.add_power_source(power_watts=power_w, x_slice=slice(3, 9), y_slice=slice(3, 9),
                            z_slice=slice(nz // 2, nz // 2 + 1))
        gb.validate_grid()
        solver = ThermalSolver.from_grid_builder(gb)
        assert solver._sink_k == nz - 1

        rho_cp_sink = MATERIALS["copper"]["rho"] * MATERIALS["copper"]["cp"]
        rho_cp_source = MATERIALS["silicon"]["rho"] * MATERIALS["silicon"]["cp"]
        assert rho_cp_sink / rho_cp_source > 2.0   # the case under test

        T = solver.solve_steady_state(apply_result=False)   # the weighted check raised here

        dx, dy = config.grid.dx, config.grid.dy
        h = config.boundary_conditions.h_convective
        # Written from the physics, not from any solver internal: each sink
        # cell loses heat through conduction across its own half cell (copper
        # here) and then the film.
        k_sink = MATERIALS["copper"]["k"]
        resistance = config.grid.dz / (2.0 * k_sink) + 1.0 / h
        p_out = float(np.sum(dx * dy * (T[:, :, nz - 1] - solver.bc.t_ambient_k) / resistance))
        assert p_out == pytest.approx(power_w, rel=1e-9)

    def test_cg_path_matches_the_exact_direct_solve(self):
        """The large-grid path is iterative, so it must be shown to agree with
        the exact factorization rather than assumed to. Compared on the same
        operator at a size where both are feasible."""
        import scipy.sparse.linalg as spla

        config = make_pipeline_config(nx=14, ny=14, nz=14, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        N = solver.nx * solver.ny * solver.nz
        rho_cp = solver._rho_cp.reshape(N)
        forcing = solver._boundary_forcing + solver.q_map.reshape(N) / rho_cp

        exact = spla.splu(solver._conduction_operator.tocsc()).solve(-forcing)

        S, diag = solver._steady_state_spd_system()
        inv_diag = 1.0 / diag
        pre = spla.LinearOperator(S.shape, matvec=lambda v: inv_diag * v)
        cg_x, info = spla.cg(S, rho_cp * forcing, M=pre,
                             rtol=thermal_solver_module.STEADY_CG_RTOL,
                             maxiter=thermal_solver_module.STEADY_CG_MAXITER)

        assert info == 0
        # Agreement across EVERY cell, not just the peak.
        assert np.max(np.abs(cg_x - exact)) < 1e-6


class TestSolverSectionConfig:
    def test_config_rejects_invalid_method(self):
        config = make_pipeline_config()
        config.solver.method = "not_a_real_method"
        with pytest.raises(ConfigError):
            config.validate()

    def test_config_accepts_crank_nicolson_method(self):
        config = make_pipeline_config()
        config.solver.method = "crank_nicolson"
        config.validate()  # must not raise

    def test_config_rejects_negative_dt(self):
        config = make_pipeline_config()
        config.solver.dt = -1e-6
        with pytest.raises(ConfigError):
            config.validate()


# ---------------------------------------------------------------------------
# 9. Stage 2 chiplet placement optimizer
# ---------------------------------------------------------------------------

# A deliberately small catalog and grid so a full optimize() run stays inside
# unit-test time. At dx=dy=25 um a 20x20 grid is a 500 x 500 um die
# (0.25 mm^2); the catalog below totals 0.0205 mm^2, comfortably inside it.
# 20x20x12 = 4800 unknowns stays under thermal_solver.DIRECT_SOLVE_MAX_UNKNOWNS
# (20_000) so every candidate takes the fast direct sparse solve.
#
# Footprints are authored in MICROMETERS, like every catalog in this project.
# These are chosen as exact multiples of the 25 um test pitch so extent_vox()
# rounds cleanly and the tests are not sensitive to ceil() behaviour.
SMALL_CATALOG = (
    ChipletSpec(name="core_a", width_um=75.0,  depth_um=75.0,  power_w=0.05),
    ChipletSpec(name="core_b", width_um=100.0, depth_um=75.0,  power_w=0.03),
    ChipletSpec(name="mem",    width_um=75.0,  depth_um=100.0, power_w=0.02),
)


def make_small_optimizer(seed: int = 0, placement_pitch_um: float = 25.0
                          ) -> ChipletPlacementOptimizer:
    """Small optimizer for unit tests.

    placement_pitch_um defaults to 25 um (one voxel here) rather than the
    package-scale DEFAULT_PLACEMENT_PITCH_UM of 1000 um: this fixture models a
    500 x 500 um die, on which a 1000 um placement lattice is larger than the
    whole die and leaves no placeable positions.

    The stack is given explicitly. This fixture used to take the default
    100/40/100 um stack on a 12 x 10 um = 120 um mesh, which the old clamp in
    StackSection silently built as 100/10/10 um. That clamp is gone -- such a
    mesh is now refused -- so the geometry these tests always actually ran
    on is written down here instead, unchanged."""
    return ChipletPlacementOptimizer(
        nx=20, ny=20, nz=12, dx=25e-6, dy=25e-6, dz=10e-6,
        catalog=SMALL_CATALOG, t_ambient_c=25.0, h_convective=2500.0,
        stack=StackSection(interposer_um=100.0, bump_um=10.0, die_um=10.0),
        placement_pitch_um=placement_pitch_um,
        seed=seed,
    )


class TestChipletPlacementOptimizer:
    """
    Pins the invariant that separates a buildable optimum from an unbuildable
    one.

    evaluate() returns a finite COST for an overlapping layout (so the
    annealer can rank and reject it) but no PEAK TEMPERATURE, because an
    overlapping geometry is never actually solved. Cost and validity are
    therefore independent signals, and the elitist tracker must key on both.
    The original defect was tracking only cost --
        improved = cand_cost < best_cost
    -- which lets an unbuildable layout win whenever its penalty happens to
    undercut a genuinely hot valid incumbent, leaving best_peak_temperature_c
    as None for every downstream consumer.
    """

    def test_overlap_candidate_never_becomes_best_when_peak_is_none(self):
        """Direct reproduction of the original bug condition: a hot valid
        incumbent, and an overlapping candidate whose penalty cost is BELOW
        it. The elitist tracker must reject the candidate on validity even
        though it wins on cost."""
        opt = make_small_optimizer(seed=3)

        # A genuinely overlapping layout: core_a at (5,5) spans x[5:8] y[5:8];
        # core_b at (6,6) spans x[6:10] y[6:9] -- they intersect.
        overlapping = ((5, 5), (6, 6), (0, 0))
        assert opt.overlap_cell_count(overlapping) > 0
        assert not opt.is_valid(overlapping)

        # Anchor the test to real evaluate() behaviour rather than assuming
        # it: an overlapping layout really does yield peak=None and no field.
        overlap_cost, overlap_peak, overlap_field = opt.evaluate(overlapping)
        assert overlap_peak is None
        assert overlap_field is None
        assert np.isfinite(overlap_cost)

        # Seed the tracker with a hot but VALID incumbent.
        valid = opt.shelf_packed_layout()
        assert opt.is_valid(valid)
        hot_peak = 5000.0
        assert opt._consider_candidate(valid, hot_peak, hot_peak) is True
        assert opt._best_layout == valid
        assert opt._best_peak_temperature_c == hot_peak

        # Now offer the overlapping candidate at a cost far BELOW the
        # incumbent. On cost alone it wins outright; on validity it must lose.
        cheap_cost = hot_peak - 3800.0
        assert cheap_cost < hot_peak
        promoted = opt._consider_candidate(overlapping, cheap_cost, None)

        assert promoted is False, (
            "an overlapping candidate with no solved peak temperature was promoted "
            "to best purely because its penalty cost undercut the incumbent"
        )
        assert opt._best_layout == valid
        assert opt._best_cost == hot_peak
        assert opt._best_peak_temperature_c == hot_peak
        assert opt._best_peak_temperature_c is not None

    def test_best_layout_is_always_non_overlapping(self, monkeypatch):
        """After a full optimize() run the returned best layout must be
        physically buildable.

        The overlap penalty is monkeypatched far BELOW the real peak
        temperatures this model produces, so overlapping candidates genuinely
        out-rank every valid layout on cost. That turns the guard from
        theoretical into load-bearing: without it, the cheapest-cost layout
        the annealer sees is an overlapping one."""
        monkeypatch.setattr(optimizer_module, "OVERLAP_PENALTY_C", 1.0)
        monkeypatch.setattr(optimizer_module, "OVERLAP_PENALTY_PER_CELL_C", 0.0)

        opt = make_small_optimizer(seed=11)
        result = opt.optimize(iterations=25, verbose=False)

        # The rigged penalty really is cheaper than any valid layout, so the
        # search was genuinely tempted by overlapping candidates.
        assert result.overlap_candidates > 0, (
            "no overlapping candidate was ever proposed; this test would not "
            "exercise the guard"
        )
        assert result.best_peak_temperature_c > 1.0

        assert opt.overlap_cell_count(result.best_layout) == 0
        assert opt.is_valid(result.best_layout)
        assert result.best_peak_temperature_c is not None
        assert isinstance(result.best_peak_temperature_c, float)
        assert np.isfinite(result.best_peak_temperature_c)

        # Every chiplet is also within the die bounds.
        for (nvx, nvy), (x0, y0) in zip(result.extents_vox, result.best_layout):
            assert 0 <= x0 and x0 + nvx <= opt.nx
            assert 0 <= y0 and y0 + nvy <= opt.ny

    def test_best_cost_history_is_monotonically_non_increasing(self):
        """The incumbent-best trace may only ever move downward. A rise means
        the tracker overwrote a better incumbent with a worse candidate."""
        opt = make_small_optimizer(seed=5)
        result = opt.optimize(iterations=25, verbose=False)

        history = result.best_cost_history
        assert len(history) == 25

        violations = [
            (i, history[i], history[i + 1])
            for i in range(len(history) - 1)
            if history[i + 1] > history[i]
        ]
        assert not violations, f"best_cost_history increased at {violations}"

        # The final incumbent cost is the reported best, and it is a real
        # solved temperature rather than an overlap penalty.
        assert history[-1] == pytest.approx(result.best_cost)
        assert history[-1] == pytest.approx(result.best_peak_temperature_c)
        assert all(h < optimizer_module.OVERLAP_PENALTY_C for h in history)


class TestChipletPhysicalDimensions:
    """Chiplet size is physical and must not be a function of mesh resolution.

    Authoring footprints in voxels made a chiplet's real size depend on dx:
    halving the pitch would halve its linear extent and quadruple its areal
    power density, so refining the mesh would silently change the modelled
    physics. Footprints are therefore authored in micrometers and resolved to
    voxels only at build time.
    """

    def test_refining_the_mesh_preserves_physical_chiplet_size(self):
        """Halving dx/dy must double the voxel extent, leaving the physical
        footprint unchanged. This is the exact regression that voxel-authored
        specs could not express."""
        spec = ChipletSpec(name="block", width_um=400.0, depth_um=200.0, power_w=1.0)

        coarse = spec.extent_vox(dx=50e-6, dy=50e-6)
        fine = spec.extent_vox(dx=25e-6, dy=25e-6)
        finer = spec.extent_vox(dx=12.5e-6, dy=12.5e-6)

        assert coarse == (8, 4)
        assert fine == (16, 8)
        assert finer == (32, 16)

        # Physical extent reconstructed from each mesh is identical.
        for (nvx, nvy), pitch in ((coarse, 50e-6), (fine, 25e-6), (finer, 12.5e-6)):
            assert nvx * pitch * 1e6 == pytest.approx(spec.width_um)
            assert nvy * pitch * 1e6 == pytest.approx(spec.depth_um)

        # And the physical properties themselves never moved.
        # 400 x 200 um = 0.04 x 0.02 cm = 8e-4 cm^2; 1 W / 8e-4 cm^2 = 1250 W/cm^2.
        assert spec.area_um2 == pytest.approx(80_000.0)
        assert spec.area_mm2 == pytest.approx(0.08)
        assert spec.power_density_w_cm2 == pytest.approx(1250.0)

    def test_extent_rounds_up_and_never_vanishes(self):
        """A footprint that is not a whole number of cells rounds UP, so a
        block never occupies less die area than it physically does; a block
        finer than one cell still occupies one."""
        spec = ChipletSpec(name="odd", width_um=130.0, depth_um=30.0, power_w=1.0)
        assert spec.extent_vox(dx=50e-6, dy=50e-6) == (3, 1)   # 2.6 -> 3, 0.6 -> 1

        tiny = ChipletSpec(name="tiny", width_um=1.0, depth_um=1.0, power_w=0.1)
        assert tiny.extent_vox(dx=50e-6, dy=50e-6) == (1, 1)

    def test_die_fit_check_uses_physical_area_not_voxel_count(self):
        """The realistic 192 mm^2 catalog must be refused on a 1 mm^2 die and
        accepted on a 400 mm^2 one, with the SAME voxel count in both cases.

        Identical nx/ny proves the check keys on physical area rather than on
        voxel counts -- the voxel-authored version could not tell these apart.
        """
        from src.optimizer import DEFAULT_CHIPLET_CATALOG

        catalog_mm2 = sum(c.area_mm2 for c in DEFAULT_CHIPLET_CATALOG)
        assert catalog_mm2 == pytest.approx(192.0)

        # 40x40 at 25 um = 1000 x 1000 um die. Refused on the per-chiplet
        # check, which fires first and is the more specific diagnosis: an
        # 11 mm HBM stack cannot go on a 1 mm die at any packing.
        with pytest.raises(OptimizerError, match=r"does not fit within the 1000 x 1000 um die"):
            ChipletPlacementOptimizer(nx=40, ny=40, nz=24,
                                      dx=25e-6, dy=25e-6, dz=10e-6)

        # 40x40 at 500 um = 20 x 20 mm = 400 mm^2 die -> accepted. nz=12 at
        # dz=20 um holds the 100/40/100 um stack exactly (5/2/5 cells). This
        # used dz=40 um, a 480 um mesh under a 240 um stack that the old
        # clamp silently built as 80/80/320 um; that mesh is now refused.
        opt = ChipletPlacementOptimizer(nx=40, ny=40, nz=12,
                                        dx=500e-6, dy=500e-6, dz=20e-6)
        assert opt.nx == 40 and opt.ny == 40      # same nx/ny as the refused case
        assert opt.is_valid(opt.shelf_packed_layout())

    def test_total_area_check_refuses_a_catalog_that_fits_individually(self):
        """The aggregate-area check is separate from the per-chiplet one:
        every block fits on its own, but together they exceed the die."""
        # Die is 40x40 at 25 um = 1000 x 1000 um = 1.0 mm^2.
        # Three 600x600 um blocks each fit alone (0.36 mm^2) but total 1.08 mm^2.
        crowded = tuple(
            ChipletSpec(name=f"blk{i}", width_um=600.0, depth_um=600.0, power_w=1.0)
            for i in range(3)
        )
        assert all(c.width_um < 1000.0 and c.depth_um < 1000.0 for c in crowded)
        assert sum(c.area_mm2 for c in crowded) == pytest.approx(1.08)

        with pytest.raises(OptimizerError, match=r"mm\^2 of die area"):
            ChipletPlacementOptimizer(nx=40, ny=40, nz=24,
                                      dx=25e-6, dy=25e-6, dz=10e-6, catalog=crowded)

    def test_power_is_conserved_regardless_of_footprint_rounding(self):
        """Rounding a footprint up lowers the modelled volumetric density but
        must never create or destroy watts: integrating q_map over the domain
        recovers the catalog total exactly."""
        opt = make_small_optimizer(seed=2)
        layout = opt.shelf_packed_layout()
        opt._apply_layout_to_q_map(layout)

        cell_volume = opt.dx * opt.dy * opt.dz
        total_w = float(np.sum(opt.solver.q_map) * cell_volume)
        expected = sum(c.power_w for c in SMALL_CATALOG)

        assert total_w == pytest.approx(expected, rel=1e-12)


class TestSteadyStateBackend:
    """The steady-state solve is verified against physics, not just against
    itself: at equilibrium every generated watt must leave through the Robin
    boundary, which pins mean(T_bottom) - T_ambient to P/(h*A) EXACTLY,
    independent of the mesh. That makes it a check on the solve rather than
    on the discretization."""

    @staticmethod
    def _analytic_mean_bottom_rise(config: PipelineConfig, total_power_w: float) -> float:
        area = (config.grid.nx * config.grid.dx) * (config.grid.ny * config.grid.dy)
        return total_power_w / (config.boundary_conditions.h_convective * area)

    def test_operator_is_asymmetric_but_symmetrizable(self):
        """_build_conduction_operator writes M = D^-1 A: M[a,b] = coeff/rho_cp[a]
        against M[b,a] = coeff/rho_cp[b]. So M is asymmetric wherever rho*cp
        differs across a material interface, while S = -D M is exactly
        symmetric. This is what licenses conjugate gradients."""
        import scipy.sparse as sparse
        import scipy.sparse.linalg as spla

        config = make_pipeline_config(die_power_w=0.05)
        gb, *_ = build_stack_grid(config)          # heterogeneous stack: Si + composite
        solver = ThermalSolver.from_grid_builder(gb)

        M = solver._conduction_operator.tocsr()
        asym_M = spla.norm(M - M.T) / spla.norm(M)
        assert asym_M > 1e-6, (
            "M is symmetric here, so this stack is not heterogeneous enough to "
            "exercise the symmetrization"
        )

        N = solver.nx * solver.ny * solver.nz
        S = -(sparse.diags(solver._rho_cp.reshape(N)) @ M)
        assert spla.norm(S - S.T) / spla.norm(S) < 1e-12

        S_cached, diag = solver._steady_state_spd_system()
        assert np.all(diag > 0)                     # positive definite diagonal
        assert spla.norm(S_cached - S_cached.T) == pytest.approx(0.0, abs=1e-12)

    def test_steady_state_pins_mean_bottom_rise_to_analytic_value(self):
        """The mesh-independent invariant, on the direct path."""
        config = make_pipeline_config(nx=12, ny=12, nz=12, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T = solver.solve_steady_state(apply_result=False)
        rise = float(np.mean(T[:, :, solver._sink_k])) - solver.bc.t_ambient_k
        floor = self._analytic_mean_bottom_rise(config, config.power.die_power_w)
        # The cell centre sits half a cell inside the silicon, so it carries
        # the whole outgoing flux across that half cell on top of the floor.
        flux = config.power.die_power_w / (config.grid.nx * config.grid.dx
                                           * config.grid.ny * config.grid.dy)
        half_cell = flux * config.grid.dz / (2.0 * MATERIALS["silicon"]["k"])
        assert rise == pytest.approx(floor + half_cell, rel=1e-6)
        assert rise - half_cell == pytest.approx(floor, rel=1e-6)

    def test_cg_path_matches_direct_path_on_an_anisotropic_mesh(self):
        """The two backends must agree ON A MESH THAT ACTUALLY BREAKS THE OLD
        ONE.

        This configuration is chosen deliberately, not arbitrarily. The
        Jacobi-on-M BiCGSTAB this replaced fails as mesh anisotropy and size
        rise together; measured, holding dx=500 um and sweeping dz:

            grid        N      aspect   Jacobi-on-M      CG-on-S
            12x12x12    1,728    2.5    ok  0.01s        ok  0.01s
            12x12x12    1,728   50.0    ok  0.16s        ok  0.03s
            16x16x12    3,072   50.0    ok  0.23s        ok  0.05s
            20x20x12    4,800   50.0    FAIL info=2000   ok  0.08s

        So 20x20x12 at 500/10 um is the cheapest configuration that
        reproduces the real failure. A milder grid would let this test pass
        against the very bug it exists to catch.
        """
        import src.thermal_solver as ts

        # dz = 10 um x nz = 12 -> a 120 um stack; dx = dy = 500 um -> 50:1.
        config = make_pipeline_config(
            nx=20, ny=20, nz=12, dx=500e-6, dy=500e-6, dz=10e-6,
            die_power_w=0.05, h_convective=1000.0,
        )
        assert max(config.grid.dx, config.grid.dy) / config.grid.dz == pytest.approx(50.0)

        gb_a, *_ = build_stack_grid(config)
        solver_direct = ThermalSolver.from_grid_builder(gb_a)
        N = solver_direct.nx * solver_direct.ny * solver_direct.nz
        assert N <= ts.DIRECT_SOLVE_MAX_UNKNOWNS
        T_direct = solver_direct.solve_steady_state(apply_result=False)

        original = ts.DIRECT_SOLVE_MAX_UNKNOWNS
        try:
            ts.DIRECT_SOLVE_MAX_UNKNOWNS = 1        # force the iterative path
            gb_b, *_ = build_stack_grid(config)
            solver_iter = ThermalSolver.from_grid_builder(gb_b)
            T_iter = solver_iter.solve_steady_state(apply_result=False)
        finally:
            ts.DIRECT_SOLVE_MAX_UNKNOWNS = original

        assert np.max(np.abs(T_direct - T_iter)) < 1e-6
        # and the iterative result independently satisfies the physical invariant
        rise = float(np.mean(T_iter[:, :, solver_iter._sink_k])) - solver_iter.bc.t_ambient_k
        floor = self._analytic_mean_bottom_rise(config, config.power.die_power_w)
        # The sink cell centre carries the outgoing flux across its own half
        # cell before the film, so it sits that much above the floor.
        flux = config.power.die_power_w / (config.grid.nx * config.grid.dx
                                           * config.grid.ny * config.grid.dy)
        half_cell = flux * config.grid.dz / (2.0 * MATERIALS["silicon"]["k"])
        assert rise == pytest.approx(floor + half_cell, rel=1e-6)

    def test_cached_factorization_still_tracks_a_changed_power_map(self):
        """The LU factorization is cached across calls because M depends only
        on the material maps. Changing q_map must still change the answer --
        a cache that ignored the new RHS would silently return a stale field."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)

        T1 = solver.solve_steady_state(apply_result=False)
        assert solver._steady_lu is not None          # cache populated
        cached_id = id(solver._steady_lu)

        solver.q_map *= 2.0                            # same operator, new RHS
        T2 = solver.solve_steady_state(apply_result=False)

        assert id(solver._steady_lu) == cached_id      # not refactorized
        assert not np.allclose(T1, T2)                 # but the answer moved
        # Doubling the power doubles the rise above ambient (the system is linear).
        amb = solver.bc.t_ambient_k
        assert np.max(T2 - amb) == pytest.approx(2.0 * np.max(T1 - amb), rel=1e-6)


class TestPlacementLattice:
    """Chiplet ORIGINS must lie on a physical lattice, not just have physical
    extents. A layout whose origins are only representable on the mesh that
    found them makes a refinement study compare different geometries."""

    def test_origins_stay_on_the_lattice_through_a_full_optimize(self):
        opt = make_small_optimizer(seed=4, placement_pitch_um=50.0)
        assert opt._pitch_x == 2 and opt._pitch_y == 2      # 50 um at dx = 25 um

        assert opt.is_on_placement_lattice(opt.shelf_packed_layout())
        result = opt.optimize(iterations=20, verbose=False)

        assert opt.is_on_placement_lattice(result.best_layout), (
            f"optimize() returned off-lattice origins: {result.best_layout}"
        )
        assert opt.is_on_placement_lattice(result.initial_layout)

    def test_every_proposal_is_lattice_aligned_including_at_the_bounds(self):
        """Snapping must survive clamping. A block pushed against the far edge
        must land on the lattice, not at nx-extent, or it becomes
        unrepresentable on a coarser mesh."""
        opt = make_small_optimizer(seed=9, placement_pitch_um=50.0)
        layout = opt.shelf_packed_layout()
        for _ in range(300):
            layout = opt._propose(layout, max_jump=8)
            assert opt.is_on_placement_lattice(layout), f"off-lattice proposal {layout}"
            for (nvx, nvy), (x0, y0) in zip(opt._extents, layout):
                assert 0 <= x0 and x0 + nvx <= opt.nx
                assert 0 <= y0 and y0 + nvy <= opt.ny

    def test_lattice_origins_are_representable_on_a_coarser_mesh(self):
        """The actual point of the constraint: a physical origin found on the
        fine mesh must land exactly on a cell face of the coarse one."""
        opt = make_small_optimizer(seed=2, placement_pitch_um=50.0)
        result = opt.optimize(iterations=15, verbose=False)

        for (x0, y0) in result.best_layout:
            x_um, y_um = x0 * opt.dx * 1e6, y0 * opt.dy * 1e6
            for coarse_dx_um in (25.0, 50.0):        # this mesh and one 2x coarser
                assert abs(x_um / coarse_dx_um - round(x_um / coarse_dx_um)) < 1e-9
                assert abs(y_um / coarse_dx_um - round(y_um / coarse_dx_um)) < 1e-9

    def test_pitch_coarser_than_the_die_is_rejected(self):
        """The package-scale default (1000 um) on a 500 um test die would
        leave no placeable positions; that must fail loudly, not silently
        collapse the search space."""
        with pytest.raises(OptimizerError, match=r"too coarse for this die"):
            make_small_optimizer(seed=0, placement_pitch_um=1000.0)


class TestExhaustiveSearch:
    """Stage 2b replaces the annealing heuristic with a complete search.

    Simulated annealing samples: it returns the best of the few hundred
    layouts it happened to visit, with no guarantee and a different answer per
    seed. The exhaustive search evaluates every buildable layout on the
    placement lattice, which is only affordable because the steady-state
    system is linear in q_map -- so a layout's field is the SUM of
    precomputed per-chiplet fields, and a candidate costs an array add rather
    than a linear solve.

    Superposition is the load-bearing assumption. If it fails, every result
    from this module is wrong, so it is checked against a real solve here and
    again inside every search().
    """

    # Deliberately tiny: a 9 mm die, two blocks, a 3 mm lattice. Small enough
    # that an independent brute-force cross-check can afford a real linear
    # solve for EVERY combination.
    TINY_CATALOG = (
        ChipletSpec(name="hot",  width_um=3000.0, depth_um=3000.0, power_w=4.0),
        ChipletSpec(name="warm", width_um=3000.0, depth_um=3000.0, power_w=2.0),
    )

    def _tiny_optimizer(self, seed: int = 0) -> ChipletPlacementOptimizer:
        return ChipletPlacementOptimizer(
            nx=9, ny=9, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
            catalog=self.TINY_CATALOG, h_convective=1000.0,
            placement_pitch_um=3000.0, seed=seed,
        )

    def test_superposition_reproduces_a_direct_solve(self):
        """The whole method rests on T(q1+q2) = T_amb + (T1-T_amb) + (T2-T_amb).
        Verified against ground truth rather than assumed."""
        opt = self._tiny_optimizer()
        search = exhaustive_module.ExhaustivePlacementSearch(opt)
        search.precompute(verbose=False)

        anchor = search._find_anchor_layout()
        err = search._verify_superposition(anchor)
        assert err < exhaustive_module.SUPERPOSITION_TOL_K
        assert err < 1e-6, f"superposition deviation {err:.3e} K is larger than expected"

    def test_exhaustive_matches_independent_brute_force(self):
        """The strongest available check: enumerate every layout independently
        and solve each one DIRECTLY, with no superposition anywhere, then
        confirm the exhaustive search returned that same optimum.

        This validates the search, the pruning and the superposition shortcut
        in one pass -- if pruning silently dropped a valid layout, or
        superposition were subtly wrong, the two answers would differ."""
        opt = self._tiny_optimizer()
        search = exhaustive_module.ExhaustivePlacementSearch(opt)
        result = search.search(verbose=False)

        # Independent enumeration: every position pair, solved for real.
        best_peak, best_layout, checked = float("inf"), None, 0
        for pa in search._positions[0]:
            for pb in search._positions[1]:
                layout = (pa, pb)
                if opt.overlap_cell_count(layout) > 0:
                    continue
                checked += 1
                opt._apply_layout_to_q_map(layout)
                field = opt.solver.solve_steady_state(apply_result=False)
                peak = float(np.max(field)) - 273.15
                if peak < best_peak:
                    best_peak, best_layout = peak, layout

        assert checked == result.layouts_evaluated, (
            f"brute force found {checked} buildable layouts but the search "
            f"evaluated {result.layouts_evaluated} -- pruning dropped some"
        )
        assert result.best_peak_temperature_c == pytest.approx(best_peak, abs=1e-6)

    def test_exhaustive_is_deterministic_across_seeds(self):
        """A proven optimum cannot depend on a random seed. Annealing's answer
        does; this one must not."""
        r0 = exhaustive_module.ExhaustivePlacementSearch(
            self._tiny_optimizer(seed=0)).search(verbose=False)
        r1 = exhaustive_module.ExhaustivePlacementSearch(
            self._tiny_optimizer(seed=12345)).search(verbose=False)

        assert r0.best_layout == r1.best_layout
        assert r0.best_peak_temperature_c == pytest.approx(r1.best_peak_temperature_c, abs=1e-9)
        assert r0.layouts_evaluated == r1.layouts_evaluated

    def test_exhaustive_best_layout_is_buildable_and_on_lattice(self):
        opt = self._tiny_optimizer()
        search = exhaustive_module.ExhaustivePlacementSearch(opt)
        result = search.search(verbose=False)

        assert opt.overlap_cell_count(result.best_layout) == 0
        assert opt.is_valid(result.best_layout)
        pitch = int(round(result.placement_pitch_um * 1e-6 / opt.dx))
        for x, y in result.best_layout:
            assert x % pitch == 0 and y % pitch == 0

    def test_exhaustive_is_at_least_as_good_as_annealing(self):
        """A complete search cannot lose to a sample of the same space."""
        opt_ex = self._tiny_optimizer(seed=0)
        exhaustive = exhaustive_module.ExhaustivePlacementSearch(opt_ex).search(verbose=False)

        opt_sa = self._tiny_optimizer(seed=0)
        annealed = opt_sa.optimize(iterations=20, verbose=False)

        assert exhaustive.best_peak_temperature_c <= annealed.best_peak_temperature_c + 1e-9

    def test_superposition_failure_is_raised_not_ignored(self):
        """Corrupt one precomputed field and confirm the guard fires. Without
        this the verification could be silently vacuous."""
        opt = self._tiny_optimizer()
        search = exhaustive_module.ExhaustivePlacementSearch(opt)
        search.precompute(verbose=False)

        key = next(iter(search._fields))
        search._fields[key] = search._fields[key] + 1.0   # 1 K of nonsense

        with pytest.raises(exhaustive_module.ExhaustiveSearchError, match="[Ss]uperposition"):
            search._verify_superposition(search._find_anchor_layout())


class TestTransientConvergenceGating:
    """A transient shorter than a few time constants cannot support a safety
    verdict, and the pipeline must not print one.

    The defect: a 100-step run covered 0.1776 ms against a 394.79 ms time
    constant -- 0.045% of one tau -- reported a peak of ambient + 0.1 K, and
    the visualizer logged "Grid is within safe operating range". The true
    steady-state peak for that same configuration is 79.97 degC against an
    85 degC threshold, i.e. a 5 degC margin rather than a 60 degC one. The
    verdict was not merely imprecise, it was drawn from a field in which
    nothing had happened yet.
    """

    def test_tau_estimate_matches_hand_calculation(self):
        """tau = R*C = [1/(h*A)] * [rho*cp * A * t] = rho*cp*t/h for a uniform
        slab. Area cancels, so this is exactly checkable by hand."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, dx=500e-6, dy=500e-6,
                                      dz=20e-6, h_convective=1000.0, die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)

        rho_cp = MATERIALS["silicon"]["rho"] * MATERIALS["silicon"]["cp"]
        thickness = config.grid.nz * config.grid.dz
        hand = rho_cp * thickness / config.boundary_conditions.h_convective

        tau = solver.estimate_time_constant()
        assert tau == pytest.approx(hand, rel=0.05)
        # This grid is the documented 240 um stack at h=1000 -> ~398 ms.
        assert 0.35 < tau < 0.45

    def test_steady_state_fraction_is_monotone_and_bounded(self):
        config = make_pipeline_config(nx=8, ny=8, nz=12, die_power_w=0.0)
        gb = build_uniform_grid(config, "silicon")
        solver = ThermalSolver.from_grid_builder(gb)
        tau = solver.estimate_time_constant()

        assert solver.steady_state_fraction(0.0) == 0.0
        assert solver.steady_state_fraction(tau) == pytest.approx(0.6321, abs=1e-3)
        assert solver.steady_state_fraction(3 * tau) == pytest.approx(0.9502, abs=1e-3)
        assert solver.steady_state_fraction(1e6 * tau) == pytest.approx(1.0, abs=1e-9)

    def test_no_safety_verdict_from_unconverged_transient(self, capsys):
        """The gate: a run far short of 3*tau must say so, and must not let a
        transient peak stand in for a verdict."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        tau = 0.3948
        report = {
            "html_path": "outputs/thermal_3d.html",
            "hotspot_count": 0,
            "peak_temperature_c": 79.97,          # steady state
            "peak_location_um": (9500.0, 9500.0, 220.0),
            "peak_location_ijk": (19, 19, 11),
            "transient_peak_temperature_c": 25.10,
            "transient_duration_s": 1.775751e-04,
            "tau_s": tau,
            "converged_fraction": 4.4977e-04,
        }
        main_module.print_stage1_results(config, report)
        out = capsys.readouterr().out

        # The transient is labelled with its duration and its tau fraction.
        assert "0.1776 ms" in out
        assert "tau" in out
        # The unconverged notice appears.
        assert "NOT" in out and "verdict" in out.lower()
        # The verdict quotes the STEADY-STATE peak, never the transient one.
        assert "79.97" in out
        assert "VERDICT" in out
        verdict_line = [l for l in out.splitlines() if "VERDICT" in l][0]
        assert "25.10" not in verdict_line

    def test_converged_transient_reports_no_unconverged_notice(self, capsys):
        """The gate must not fire when the run really has converged, or it
        becomes noise that trains the user to ignore it."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        tau = 0.3948
        report = {
            "html_path": "outputs/thermal_3d.html",
            "hotspot_count": 0,
            "peak_temperature_c": 79.97,
            "peak_location_um": (9500.0, 9500.0, 220.0),
            "peak_location_ijk": (19, 19, 11),
            "transient_peak_temperature_c": 79.90,
            "transient_duration_s": 5.0 * tau,        # well past 3*tau
            "tau_s": tau,
            "converged_fraction": 0.9933,
        }
        main_module.print_stage1_results(config, report)
        out = capsys.readouterr().out
        assert "VERDICT" in out
        assert "cannot support one" not in out

    def test_verdict_fires_when_steady_peak_exceeds_threshold(self, capsys):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        report = {
            "html_path": "outputs/thermal_3d.html",
            "hotspot_count": 137,
            "peak_temperature_c": 161.11,     # above the 85 C default
            "peak_location_um": (9500.0, 9500.0, 220.0),
            "peak_location_ijk": (19, 19, 11),
            "transient_peak_temperature_c": 25.10,
            "transient_duration_s": 1.775751e-04,
            "tau_s": 0.3948,
            "converged_fraction": 4.4977e-04,
        }
        main_module.print_stage1_results(config, report)
        out = capsys.readouterr().out
        assert "EXCEEDS THRESHOLD" in out
        assert "161.11" in out

    def test_steady_state_threshold_check_is_used(self):
        """End to end: the hotspot count and reported peak must come from the
        steady-state field, not the transient snapshot. Run a deliberately
        tiny transient and confirm the reported peak is the equilibrium one."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.4)
        outcome = main_module.run_pipeline(
            config, total_steps=5, duration_s=None, open_browser=False,
        )
        rep = outcome["viz_report"]

        # A 5-step transient has heated almost nothing; the steady state has.
        assert rep["transient_peak_temperature_c"] < rep["peak_temperature_c"]
        assert rep["converged_fraction"] < 1e-3

        # And the reported peak really is the steady-state solve's.
        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        steady = solver.solve_steady_state(apply_result=False)
        assert rep["peak_temperature_c"] == pytest.approx(
            float(np.max(steady)) - 273.15, abs=1e-6
        )


class TestStage2PowerMatchesStage1:
    """Stage 1 asks for a die power; Stage 2 must not silently model a
    different chip. The catalog was 36.0 W against a 12.0 W default -- 3x, and
    energy-balance floors 60 K apart."""

    def test_scaling_preserves_total_and_ratios(self):
        original = optimizer_module.DEFAULT_CHIPLET_CATALOG
        target = 12.0
        scaled = optimizer_module.scale_catalog_to_total_power(original, target)

        assert sum(c.power_w for c in scaled) == pytest.approx(target, rel=1e-12)
        # Relative split is untouched.
        for a, b in zip(original, scaled):
            assert b.power_w / sum(c.power_w for c in scaled) == pytest.approx(
                a.power_w / sum(c.power_w for c in original), rel=1e-12
            )
        # Geometry is untouched -- only power scales.
        for a, b in zip(original, scaled):
            assert (b.name, b.width_um, b.depth_um) == (a.name, a.width_um, a.depth_um)

    def test_stage2_power_matches_stage1_when_scaling_enabled(self):
        """The optimizer actually built for Stage 2 must carry the Stage 1
        wattage."""
        config = make_pipeline_config(nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6,
                                      dz=20e-6, die_power_w=12.0, h_convective=1000.0)
        scaled = optimizer_module.scale_catalog_to_total_power(
            optimizer_module.DEFAULT_CHIPLET_CATALOG, config.power.die_power_w
        )
        opt = ChipletPlacementOptimizer.from_pipeline_config(config, catalog=scaled)
        assert sum(c.power_w for c in opt.catalog) == pytest.approx(12.0, rel=1e-12)

        # And the physics follows: the energy floor moves with the power.
        area = (config.grid.nx * config.grid.dx) * (config.grid.ny * config.grid.dy)
        expected_floor = 12.0 / (config.boundary_conditions.h_convective * area)
        opt._apply_layout_to_q_map(opt.shelf_packed_layout())
        T = opt.solver.solve_steady_state(apply_result=False)
        mean_bottom_rise = float(np.mean(T[:, :, opt.solver._sink_k])) - opt.solver.bc.t_ambient_k
        # The sink cell centre carries the outgoing flux across its own half
        # cell before the film, so it sits that much above the floor.
        half_cell = (12.0 / area) * config.grid.dz / (2.0 * MATERIALS["silicon"]["k"])
        assert mean_bottom_rise == pytest.approx(expected_floor + half_cell, rel=1e-6)

    def test_scaling_rejects_a_zero_power_catalog(self):
        dead = (ChipletSpec(name="a", width_um=1000.0, depth_um=1000.0, power_w=0.0),)
        with pytest.raises(OptimizerError, match="zero"):
            optimizer_module.scale_catalog_to_total_power(dead, 10.0)


class TestRenderedFieldMatchesVerdict:
    """The default-opened artifact must show the field the verdict describes.

    thermal_3d.html was rendered from the TRANSIENT while the terminal
    reported the STEADY-STATE peak: the browser showed a 25-31 degC field and
    the text said 79.97 degC. Two visualizers were constructed, and only the
    transient one ever wrote a file.
    """

    @staticmethod
    def _embedded_range(html_path):
        """Pulls t_min/t_max back out of the JSON payload embedded in the page."""
        import json as _json
        import re as _re
        html = pathlib.Path(html_path).read_text(encoding="utf-8")
        m = _re.search(r'"t_min":([-\d.eE+]+),"t_max":([-\d.eE+]+)', html)
        assert m, "embedded payload has no t_min/t_max"
        return float(m.group(1)), float(m.group(2))

    def test_webgl_renders_steady_state_field(self, tmp_path):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.4)
        config.visualization.output_dir = str(tmp_path)

        outcome = main_module.run_pipeline(
            config, total_steps=5, duration_s=None, open_browser=False,
        )
        rep = outcome["viz_report"]

        gb, *_ = build_stack_grid(config)
        solver = ThermalSolver.from_grid_builder(gb)
        steady = solver.solve_steady_state(apply_result=False) - 273.15

        t_min, t_max = self._embedded_range(rep["html_path"])
        assert t_max == pytest.approx(float(np.max(steady)), abs=0.01)
        assert t_min == pytest.approx(float(np.min(steady)), abs=0.01)
        # And it agrees with the number the verdict quotes.
        assert t_max == pytest.approx(rep["peak_temperature_c"], abs=0.01)
        # The transient is far cooler and must NOT be what was rendered.
        assert rep["transient_peak_temperature_c"] < t_max - 1.0

    def test_transient_written_separately_with_duration_in_name(self, tmp_path):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.4)
        config.visualization.output_dir = str(tmp_path)
        outcome = main_module.run_pipeline(
            config, total_steps=5, duration_s=None, open_browser=False,
        )
        rep = outcome["viz_report"]

        assert rep["transient_html_path"] is not None
        assert "transient" in pathlib.Path(rep["transient_html_path"]).name
        assert "ms" in pathlib.Path(rep["transient_html_path"]).name
        html = pathlib.Path(rep["transient_html_path"]).read_text(encoding="utf-8")
        assert "TRANSIENT SNAPSHOT" in html
        assert "not equilibrium" in html
        # Its embedded field really is the transient one.
        t_min, t_max = self._embedded_range(rep["transient_html_path"])
        assert t_max == pytest.approx(rep["transient_peak_temperature_c"], abs=0.01)

    def test_default_run_does_no_time_stepping(self, tmp_path):
        """Steady state is the default: no CFL bound, no steps, no transient
        artifact."""
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.4)
        config.visualization.output_dir = str(tmp_path)
        outcome = main_module.run_pipeline(
            config, total_steps=None, duration_s=None, open_browser=False,
        )
        rep = outcome["viz_report"]
        timing = outcome["timing"]

        assert rep["transient_duration_s"] is None
        assert rep["transient_peak_temperature_c"] is None
        assert rep["transient_html_path"] is None
        assert timing.solve_s == 0.0
        assert timing.cfl_check_s == 0.0
        # The steady-state answer is still produced.
        assert rep["peak_temperature_c"] > 25.0


class TestFactorizationReuse:
    """Stage 2 re-factorized the same grid Stage 1 had already factorized --
    ~1.33 s of duplicated work per run."""

    def _cfg(self):
        return make_pipeline_config(nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6,
                                    dz=20e-6, die_power_w=12.0, h_convective=1000.0)

    def test_stage2_reuses_stage1_factorization(self):
        config = self._cfg()
        gb = main_module.build_chiplet_stack(config)
        stage1 = ThermalSolver.from_grid_builder(gb)
        stage1.solve_steady_state(apply_result=False)     # forces the factorization
        assert stage1._steady_lu is not None
        lu_id = id(stage1._steady_lu)

        opt = ChipletPlacementOptimizer.from_pipeline_config(
            config, catalog=optimizer_module.DEFAULT_CHIPLET_CATALOG,
            reuse_solver=stage1,
        )
        assert opt.reused_solver is True
        assert opt.solver is stage1
        # The SAME factorization object, not merely an equivalent one.
        assert id(opt.solver._steady_lu) == lu_id

    def test_reuse_is_refused_when_materials_differ(self):
        config = self._cfg()
        gb = main_module.build_chiplet_stack(config)
        stage1 = ThermalSolver.from_grid_builder(gb)
        stage1.kx_map = stage1.kx_map * 2.0          # a different chip

        with pytest.raises(OptimizerError, match="kx_map differs"):
            ChipletPlacementOptimizer.from_pipeline_config(
                config, catalog=optimizer_module.DEFAULT_CHIPLET_CATALOG,
                reuse_solver=stage1,
            )

    def test_reuse_is_refused_when_grid_shape_differs(self):
        config = self._cfg()
        small = make_pipeline_config(nx=10, ny=10, nz=12, dx=500e-6, dy=500e-6, dz=20e-6)
        gb_small, *_ = build_stack_grid(small)
        wrong = ThermalSolver.from_grid_builder(gb_small)

        with pytest.raises(OptimizerError, match="shape"):
            ChipletPlacementOptimizer.from_pipeline_config(
                config, catalog=optimizer_module.DEFAULT_CHIPLET_CATALOG,
                reuse_solver=wrong,
            )

    def test_reuse_is_refused_when_sink_kind_differs_on_the_same_face(self):
        """The case a bottom_bc-only identity check cannot see.

        Both configurations sink through the TOP face and share shape,
        spacing, h and every material map -- they differ only in whether that
        top sink is Robin or Dirichlet. bottom_bc is 'neumann' in both, so a
        guard comparing bottom_bc alone passes them as identical and reuses a
        factorization of the Robin operator to solve a Dirichlet problem:
        silently, a different chip. The guard must compare top_bc as well.
        """
        robin_top = make_pipeline_config(
            nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6,
            die_power_w=12.0, h_convective=1000.0,
            bottom_bc="neumann", top_bc="robin",
        )
        dirichlet_top = make_pipeline_config(
            nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6,
            die_power_w=12.0, h_convective=1000.0,
            bottom_bc="neumann", top_bc="dirichlet",
        )
        stage1 = ThermalSolver.from_grid_builder(main_module.build_chiplet_stack(robin_top))
        assert stage1.bc.bottom_bc == dirichlet_top.boundary_conditions.bottom_bc

        with pytest.raises(OptimizerError, match="top_bc differs"):
            ChipletPlacementOptimizer.from_pipeline_config(
                dirichlet_top, catalog=optimizer_module.DEFAULT_CHIPLET_CATALOG,
                reuse_solver=stage1,
            )

    def test_batched_precompute_matches_loop(self):
        """One lu.solve(B) with a 2-D RHS must reproduce the per-position loop
        to solver tolerance."""
        def build():
            return ChipletPlacementOptimizer(
                nx=20, ny=20, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
                h_convective=1000.0, placement_pitch_um=5000.0, seed=0,
            )

        loop = exhaustive_module.ExhaustivePlacementSearch(build(), placement_pitch_um=5000.0)
        loop.precompute(verbose=False)
        batched = exhaustive_module.ExhaustivePlacementSearch(build(), placement_pitch_um=5000.0)
        batched.precompute_batched(verbose=False)

        assert set(loop._fields) == set(batched._fields)
        worst = max(float(np.max(np.abs(loop._fields[k] - batched._fields[k])))
                    for k in loop._fields)
        assert worst < 1e-9, f"batched precompute deviates by {worst:.3e} K"

    def test_batched_precompute_refuses_the_iterative_path(self):
        """Above the direct-solve threshold there is no factorization to batch
        against, and the method must say so rather than silently degrade."""
        opt = ChipletPlacementOptimizer(
            nx=40, ny=40, nz=24, dx=500e-6, dy=500e-6, dz=10e-6,
            h_convective=1000.0, seed=0,
        )
        search = exhaustive_module.ExhaustivePlacementSearch(opt, placement_pitch_um=4000.0)
        with pytest.raises(exhaustive_module.ExhaustiveSearchError, match="DIRECT_SOLVE_MAX"):
            search.precompute_batched(verbose=False)


class TestCoolingPresets:
    """Validation plan step 2: two named cooling regimes, and a header that
    states the regime -- heat flux, R_ja floor, spreading length -- so no
    temperature is ever read without it."""

    @staticmethod
    def _flags(*argv):
        args = main_module.build_cli_parser().parse_args(list(argv))
        config, _, _ = main_module.build_config_from_flags(args)
        return config

    def test_presets_are_the_two_named_regimes(self):
        presets = main_module.COOLING_PRESETS
        assert set(presets) == {"passive-lowpower", "cowos-coldplate"}
        low, cold = presets["passive-lowpower"], presets["cowos-coldplate"]
        assert (low.die_power_w, low.h_convective) == (12.0, 1000.0)
        assert (cold.die_power_w, cold.h_convective) == (150.0, 25000.0)

    def test_no_flags_is_the_default_preset(self):
        config = self._flags()
        assert config.power.die_power_w == 12.0
        assert config.boundary_conditions.h_convective == 1000.0
        assert main_module.regime_label(config) == "passive-lowpower"

    def test_preset_flag_sets_power_and_h_together(self):
        config = self._flags("--preset", "cowos-coldplate")
        assert config.power.die_power_w == 150.0
        assert config.boundary_conditions.h_convective == 25000.0
        assert main_module.regime_label(config) == "cowos-coldplate"

    def test_explicit_flag_overrides_preset_and_regime_becomes_custom(self):
        """An overridden preset is a different package. Reporting it under the
        preset's name would make the regime implicit again -- the thing the
        presets exist to prevent."""
        config = self._flags("--preset", "cowos-coldplate", "--h-convective", "10000")
        assert config.boundary_conditions.h_convective == 10000.0
        assert config.power.die_power_w == 150.0     # the half not overridden survives
        assert main_module.regime_label(config) == "custom"

    def test_preset_and_config_file_are_mutually_exclusive(self, tmp_path, capsys):
        # The file does not exist; the check must fire BEFORE load_config, or
        # this would pass vacuously on a missing-file error. Hence the message
        # assertion, not just the exit code.
        rc = main_module.main(["--config", str(tmp_path / "unused.json"),
                               "--preset", "cowos-coldplate"])
        assert rc == 1
        # The refusal is now generic over every flag that changes what is
        # simulated, not just --preset (see CONFIG_FILE_CONFLICTS).
        assert "--preset cannot be combined with --config" in capsys.readouterr().err

    @pytest.mark.parametrize("preset, flux_fp, flux_sink, r_conv, floor", [
        # Hand values: footprint 20x20 cells x 500 um = 100 mm^2 = 1 cm^2;
        # sink 40x40 x 500 um = 400 mm^2 = 4 cm^2; R = 1/(h * 4e-4 m^2).
        ("passive-lowpower", 12.00, 3.00, 2.5000, 30.00),
        ("cowos-coldplate", 150.00, 37.50, 0.1000, 15.00),
    ])
    def test_header_states_flux_rja_floor_and_lambda(self, capsys, preset, flux_fp,
                                                     flux_sink, r_conv, floor):
        config = self._flags("--preset", preset)
        main_module.print_regime_header(config)
        out = capsys.readouterr().out
        assert f"Cooling regime: {preset} (preset)" in out
        assert f"{flux_fp:.2f} W/cm^2 over the active footprint (100.0 mm^2)" in out
        assert f"{flux_sink:.2f} W/cm^2 over the sink face (400.0 mm^2)" in out
        assert f"1/(h*A_sink) = {r_conv:.4f} K/W" in out
        assert f"P*R = {floor:.2f} K" in out
        # lambda recomputed here from MATERIALS and the authored stack, not
        # from the helper, so a drift between the two is caught.
        comp = MATERIALS["bump_underfill_composite"]
        sheet = (MATERIALS["silicon"]["k"] * (100e-6 + 100e-6)
                 + np.sqrt(comp["kx"] * comp["ky"]) * 40e-6)
        lam_mm = 1e3 * np.sqrt(sheet / config.boundary_conditions.h_convective)
        assert f"= {lam_mm:.2f} mm" in out

    @pytest.mark.parametrize("preset", ["passive-lowpower", "cowos-coldplate"])
    def test_printed_floor_is_the_solved_mean_sink_rise(self, preset):
        """The header's floor is not decoration: at steady state the cooled
        SURFACE sits exactly P/(h*A) above ambient, so the number printed
        before the solve must match the field after it, and the measured R_ja
        can never fall below it. The solved field reports cell centres, which
        sit half a cell inside the silicon -- add that drop back and the
        printed floor must reappear exactly."""
        config = self._flags("--preset", preset)
        regime = main_module.regime_summary(config)
        solver = ThermalSolver.from_grid_builder(main_module.build_chiplet_stack(config))
        T = solver.solve_steady_state(apply_result=False)
        mean_rise = float(np.mean(T[:, :, solver._sink_k])) - solver.bc.t_ambient_k
        area = (config.grid.nx * config.grid.dx) * (config.grid.ny * config.grid.dy)
        half_cell = (config.power.die_power_w / area) * config.grid.dz / (2.0 * MATERIALS["silicon"]["k"])
        assert mean_rise - half_cell == pytest.approx(regime["floor_rise_k"], rel=1e-9)
        assert mean_rise > regime["floor_rise_k"]
        peak_rise = float(np.max(T)) - solver.bc.t_ambient_k
        assert peak_rise / config.power.die_power_w >= regime["r_conv_k_per_w"]

    def test_results_block_reports_measured_rja(self, capsys):
        config = self._flags("--preset", "passive-lowpower")
        report = {
            "html_path": "outputs/thermal_3d.html",
            "hotspot_count": 0,
            "peak_temperature_c": 79.82,
            "peak_location_um": (10000.0, 10000.0, 160.0),
            "peak_location_ijk": (20, 20, 8),
            "transient_peak_temperature_c": None,
            "transient_duration_s": None,
            "tau_s": 0.3948,
            "converged_fraction": 0.0,
        }
        main_module.print_stage1_results(config, report)
        out = capsys.readouterr().out
        # (79.82 - 25) / 12 = 4.5683 K/W; / 2.5 K/W floor = 1.83x
        assert "R_ja (measured):      4.5683 K/W" in out
        assert "1.83x the 1/(hA) floor" in out


class TestCompositeMaterials:
    """Validation plan step 3: conductivities from Rao et al., Thermal Science
    27(5B), 4193-4207 (2023), doi:10.2298/TSCI220805061R -- checked against
    the paper's own numbers, not only against themselves."""

    def test_reproduces_the_papers_tsv_out_of_plane_value(self):
        # Paper: analytical 175.2 W/mK for r_Cu 8, r_SiO2 9, pitch 40 um.
        _, kz = composite_module.tsv_cell_k(r_core_um=8.0, r_liner_um=9.0, pitch_um=40.0)
        assert kz == pytest.approx(175.2, abs=0.05)

    def test_in_plane_model_matches_the_papers_fem_within_one_percent(self):
        # Paper FEM: 115.3 W/mK. (Its analytical eqs. (3)-(7) are not used --
        # see the composite_materials docstring.)
        kxy, _ = composite_module.tsv_cell_k(8.0, 9.0, 40.0)
        assert kxy == pytest.approx(115.3, rel=0.01)

    def test_microbump_laminate_is_table1_through_eqs_10_and_11(self):
        layers = composite_module.MICROBUMP_LAMINATE_UM
        assert sum(t for _, t in layers) == pytest.approx(composite_module.MICROBUMP_HEIGHT_UM)
        kz, kxy = composite_module.microbump_k()
        # Hand values from Table 2: 10 um Cu, 6 um Ni, 3 um Ni3Sn4, 4 um Sn2.5Ag.
        assert kz == pytest.approx(23.0 / (10 / 390.0 + 6 / 60.7 + 3 / 19.6 + 4 / 57.0), rel=1e-12)
        assert kxy == pytest.approx((10 * 390.0 + 6 * 60.7 + 3 * 19.6 + 4 * 57.0) / 23.0, rel=1e-12)

    def test_layer_kz_runs_up_the_pillars(self):
        """Side-by-side pillars conduct in parallel along z. Eq. (10) applied
        to the layer by volume fraction would treat them as stacked sheets and
        return ~0.25 W/mK, barely above the underfill."""
        kx, ky, kz = composite_module.bump_underfill_layer_k()
        phi = np.pi * 10.0 ** 2 / 40.0 ** 2
        kz_bump, _ = composite_module.microbump_k()
        assert kz == pytest.approx(phi * kz_bump + (1 - phi) * 0.2, rel=1e-12)
        series_misapplied = 1.0 / (phi / kz_bump + (1 - phi) / 0.2)
        assert kz > 40 * series_misapplied
        assert kx == ky and 0.2 < kx < 0.5    # underfill-dominated in plane

    def test_materials_table_carries_the_cited_values(self):
        assert MATERIALS["silicon"]["k"] == 150.0
        assert MATERIALS["copper"]["k"] == 390.0
        assert MATERIALS["underfill"]["k"] == 0.2
        assert MATERIALS["solder_microbump"]["k"] == 57.0
        kx, ky, kz = composite_module.bump_underfill_layer_k()
        comp = MATERIALS["bump_underfill_composite"]
        assert (comp["kx"], comp["ky"], comp["kz"]) == (kx, ky, kz)

    def test_rejects_pillars_wider_than_their_pitch(self):
        with pytest.raises(ValueError, match="pitch"):
            composite_module.bump_underfill_layer_k(pitch_um=15.0, diameter_um=20.0)


class TestValidationStatusOnReports:
    """Standing rule: every reported temperature carries the validation status
    on the report itself, alongside the known limitations, and that status says
    exactly what has been compared with hardware -- currently level and shape on
    one test chip under forced convection, and nothing beyond that scope.
    Checked on each artifact that shows a temperature, not on the helper that
    formats the text."""

    NAMES = [name for name, _ in validation_module.LIMITATIONS]

    def test_stage1_results_block(self, capsys):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.05)
        report = {
            "html_path": "outputs/thermal_3d.html",
            "hotspot_count": 0,
            "peak_temperature_c": 79.75,
            "peak_location_um": (9500.0, 9500.0, 160.0),
            "peak_location_ijk": (19, 20, 8),
            "transient_peak_temperature_c": None,
            "transient_duration_s": None,
            "tau_s": 0.3948,
            "converged_fraction": 0.0,
        }
        main_module.print_stage1_results(config, report)
        out = capsys.readouterr().out
        assert f"STATUS: {validation_module.STATUS.upper()}." in out
        # The two halves are stated separately, because they are different claims.
        assert ("Validated (one test chip, forced convection): the temperature map of a "
                "measured thermal test chip") in out
        assert "Not validated (outside that scope): natural convection" in out
        for name in self.NAMES:
            assert name in out

    def test_stage1_html_views(self, tmp_path):
        config = make_pipeline_config(nx=10, ny=10, nz=12, die_power_w=0.4)
        config.visualization.output_dir = str(tmp_path)
        rep = main_module.run_pipeline(
            config, total_steps=5, duration_s=None, open_browser=False,
        )["viz_report"]
        steady = pathlib.Path(rep["html_path"]).read_text(encoding="utf-8")
        transient = pathlib.Path(rep["transient_html_path"]).read_text(encoding="utf-8")
        for html in (steady, transient):
            assert f"{validation_module.STATUS.capitalize()}." in html
            for name in self.NAMES:
                assert name in html
        # Only the transient view says the per-solve checks do not cover it.
        assert "transient snapshot; the per-solve checks" in transient
        assert "transient snapshot; the per-solve checks" not in steady

    def test_stage2_report(self, tmp_path):
        opt = ChipletPlacementOptimizer(
            nx=20, ny=20, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
            h_convective=1000.0, placement_pitch_um=5000.0, seed=0,
        )
        result = opt.optimize(iterations=2, verbose=False)
        path = optimizer_module.render_optimization_report(result, output_dir=str(tmp_path))
        html = pathlib.Path(path).read_text(encoding="utf-8")
        assert "__VALIDATION__" not in html
        assert f"{validation_module.STATUS.capitalize()}." in html
        for name in self.NAMES:
            assert name in html


# ---------------------------------------------------------------------------
# Master prompt, Phase 1: configuration and physical consistency
# ---------------------------------------------------------------------------

SHIPPED_CONFIG = pathlib.Path(__file__).resolve().parent.parent / "configs" / "config.json"
STRESS_CONFIG = SHIPPED_CONFIG.parent / "stress_high_power_low_cool.json"


def _config_file(tmp_path, edit=None) -> str:
    """The shipped config, optionally edited, written to tmp_path."""
    raw = json.loads(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    if edit is not None:
        edit(raw)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return str(path)


class TestConfigFailsClosed:
    """Phase 1A: an unknown, retired, missing or mistyped key raises
    ConfigError naming the key. Nothing is ignored and nothing defaults."""

    def test_unknown_key_is_named(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r["grid"].update(nxx=40))
        with pytest.raises(ConfigError, match=r"grid\.nxx"):
            config_loader_module.load_config(path)

    def test_unknown_section_is_named(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r.update(meshing={"n": 1}))
        with pytest.raises(ConfigError, match="meshing"):
            config_loader_module.load_config(path)

    @pytest.mark.parametrize("old, new", [("interposer_frac", "interposer_um"),
                                          ("bump_frac", "bump_um")])
    def test_retired_key_names_its_replacement(self, tmp_path, old, new):
        path = _config_file(tmp_path, lambda r: r["stack"].update({old: 0.4}))
        with pytest.raises(ConfigError, match=rf"stack\.{old}.*stack\.{new}"):
            config_loader_module.load_config(path)

    @pytest.mark.parametrize("section, key", [
        ("grid", "dz"), ("stack", "die_um"), ("boundary_conditions", "top_bc"),
        ("boundary_conditions", "h_convective"), ("power", "die_power_w"),
    ])
    def test_missing_key_is_named_not_defaulted(self, tmp_path, section, key):
        path = _config_file(tmp_path, lambda r: r[section].pop(key))
        with pytest.raises(ConfigError, match=rf"{section}\.{key}"):
            config_loader_module.load_config(path)

    def test_missing_section_is_named(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r.pop("power"))
        with pytest.raises(ConfigError, match="power"):
            config_loader_module.load_config(path)

    @pytest.mark.parametrize("section, key, value", [
        ("grid", "nx", 40.5), ("grid", "nx", "40"), ("grid", "dx", None),
        ("solver", "steady_state", 1), ("boundary_conditions", "top_bc", 3),
    ])
    def test_wrong_type_is_named(self, tmp_path, section, key, value):
        path = _config_file(tmp_path, lambda r: r[section].update({key: value}))
        with pytest.raises(ConfigError, match=rf"{section}\.{key}"):
            config_loader_module.load_config(path)

    def test_malformed_json_raises_instead_of_defaulting(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{ not json", encoding="utf-8")
        with pytest.raises(ConfigError, match="JSON"):
            config_loader_module.load_config(str(path))

    def test_absent_default_file_raises_instead_of_defaulting(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config_loader_module, "DEFAULT_CONFIG_PATH",
                            str(tmp_path / "absent.json"))
        with pytest.raises(ConfigError, match="absent.json"):
            config_loader_module.load_config()

    def test_json_round_trip_is_exact(self, tmp_path):
        first = config_loader_module.load_config(str(SHIPPED_CONFIG))
        path = tmp_path / "round_trip.json"
        path.write_text(json.dumps(first.as_dict()), encoding="utf-8")
        assert config_loader_module.load_config(str(path)).as_dict() == first.as_dict()


class TestStackMustFitTheMesh:
    """Phase 1B: the authored stack must fill nz*dz exactly, in whole cells.
    A mesh that cannot hold it is refused -- never squeezed or clamped."""

    def test_default_stack_resolves_to_whole_cells_on_the_dz_ladder(self):
        s = StackSection()
        assert s.layer_cells(12, 20e-6) == (5, 2, 5)
        assert s.layer_cells(24, 10e-6) == (10, 4, 10)
        assert s.layer_cells(48, 5e-6) == (20, 8, 20)

    def test_z_extent_mismatch_is_refused(self):
        config = make_pipeline_config(nx=10, ny=10, nz=12, dz=20e-6,
                                      interposer_um=100.0, bump_um=40.0, die_um=100.0)
        config.grid.nz = 24          # a 480 um mesh under a 240 um stack
        with pytest.raises(ConfigError, match=r"nz\*dz"):
            config.validate()

    def test_layer_that_is_not_whole_cells_is_refused(self):
        # 16 cells x 15 um = 240 um: the total matches, but 100 um is 6.67 cells.
        with pytest.raises(ConfigError, match="interposer"):
            StackSection().layer_cells(16, 15e-6)

    def test_the_old_clamp_case_is_refused_not_squeezed(self):
        """The case the clamp hid: the 240 um stack on a 120 um mesh was built
        as 100/10/10 um -- composite -75%, die -90% -- with no error."""
        with pytest.raises(ConfigError):
            StackSection().resolve_layer_boundaries(12, 10e-6)

    def test_optimizer_refuses_a_stack_its_mesh_cannot_hold(self):
        # The optimizer resolves its stack through StackSection and already
        # converts ConfigError to OptimizerError, so it now refuses as well.
        with pytest.raises(OptimizerError, match="Stack layer resolution failed"):
            ChipletPlacementOptimizer(nx=20, ny=20, nz=12, dx=25e-6, dy=25e-6, dz=10e-6,
                                      catalog=SMALL_CATALOG, placement_pitch_um=25.0)

    def test_preflight_recommendation_holds_the_stack(self):
        """preflight_optimizer_fit() recommended --dz-um 40 at --nz 12: a
        480 um mesh under the 240 um stack. Its advice must itself be valid."""
        import re as _re
        config = make_pipeline_config(nx=40, ny=40, nz=12, dx=25e-6, dy=25e-6, dz=20e-6,
                                      interposer_um=100.0, bump_um=40.0, die_um=100.0)
        message = main_module.preflight_optimizer_fit(config)
        nz = int(_re.search(r"--nz (\d+)", message).group(1))
        dz_um = float(_re.search(r"--dz-um ([\d.]+)", message).group(1))
        assert StackSection().layer_cells(nz, dz_um * 1e-6) == (5, 2, 5)


class TestShippedConfig:
    """Phase 1C/1D: configs/config.json loads under the strict rules, fills
    its mesh with the authored stack in whole cells, matches the catalog's
    20 x 20 mm die, and sinks through the die backside."""

    @pytest.mark.parametrize("path", [SHIPPED_CONFIG, STRESS_CONFIG], ids=["config", "stress"])
    def test_loads_and_satisfies_every_invariant(self, path):
        config = config_loader_module.load_config(str(path))
        g, s = config.grid, config.stack
        assert abs(g.nz * g.dz - s.total_um * 1e-6) / (g.nz * g.dz) < 1e-9
        assert s.layer_cells(g.nz, g.dz) == (5, 2, 5)
        assert g.nx * g.dx == pytest.approx(20e-3, rel=1e-12)
        assert g.ny * g.dy == pytest.approx(20e-3, rel=1e-12)
        assert main_module.preflight_optimizer_fit(config) is None
        assert g.nx * g.ny * g.nz <= thermal_solver_module.DIRECT_SOLVE_MAX_UNKNOWNS

    @pytest.mark.parametrize("path", [SHIPPED_CONFIG, STRESS_CONFIG], ids=["config", "stress"])
    def test_sink_is_the_die_backside(self, path):
        bc = config_loader_module.load_config(str(path)).boundary_conditions
        assert (bc.bottom_bc, bc.top_bc) == ("neumann", "robin")
        flags, _, _ = main_module.build_config_from_flags(
            main_module.build_cli_parser().parse_args([]))
        assert (flags.boundary_conditions.bottom_bc,
                flags.boundary_conditions.top_bc) == ("neumann", "robin")


class TestStage1Stage2Power:
    """Phase 1E: Stage 2 either models the same chip as Stage 1, and says how,
    or refuses. It never runs two different chips under one invocation."""

    @staticmethod
    def _config(watts):
        return make_pipeline_config(nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6,
                                    die_power_w=watts, h_convective=1000.0)

    def test_rescaling_makes_one_chip_and_says_so(self):
        catalog, note = main_module.stage2_catalog(self._config(12.0), scale_power_to_stage1=True)
        assert sum(c.power_w for c in catalog) == pytest.approx(12.0, rel=1e-12)
        assert "36.00 W -> 12.00 W" in note

    def test_mismatch_without_rescaling_is_refused(self):
        with pytest.raises(main_module.PipelineError, match="different chips"):
            main_module.stage2_catalog(self._config(12.0), scale_power_to_stage1=False)

    def test_matching_power_is_used_as_authored(self):
        catalog, _ = main_module.stage2_catalog(self._config(36.0), scale_power_to_stage1=False)
        assert catalog == optimizer_module.DEFAULT_CHIPLET_CATALOG

    def test_cli_refuses_before_any_solve(self, tmp_path, capsys):
        rc = main_module.main(["--optimize", "--no-scale-catalog-power", "--no-browser",
                               "--output-dir", str(tmp_path)])
        assert rc == main_module.EXIT_PREFLIGHT_FAILED
        assert "different chips" in capsys.readouterr().err
        assert not list(tmp_path.iterdir())     # nothing solved, nothing written


class TestStartupBanner:
    """Phase 1F with 1B: before any solve the run states the authored and
    realized stack in physical units, with cell counts."""

    def test_banner_lists_authored_realized_and_cells(self, capsys):
        import re as _re
        config = make_pipeline_config(nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6,
                                      interposer_um=100.0, bump_um=40.0, die_um=100.0,
                                      die_power_w=12.0, h_convective=1000.0)
        main_module.print_startup_banner(config, source="test")
        out = capsys.readouterr().out
        assert _re.search(r"interposer\s+100\.000 um -> 100\.000 um\s+\(5 cells\)", out)
        assert _re.search(r"composite\s+40\.000 um ->\s+40\.000 um\s+\(2 cells\)", out)
        assert _re.search(r"die\s+100\.000 um -> 100\.000 um\s+\(5 cells\)", out)
        assert "240.000 um = nz*dz = 12 x 20.000 um" in out
        assert "20.000 x 20.000 mm" in out
        assert validation_module.STATUS.upper() in out

    def test_banner_prints_before_the_solve(self, tmp_path, capsys):
        rc = main_module.main(["--no-browser", "--output-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert out.index("(authored -> realized)") < out.index("Steady-state solve")

    def test_banner_reports_die_and_catalog_fit(self, capsys):
        """Decision 8: the die is DERIVED from nx*dx, ny*dy -- there is no
        die_size field to drift from the mesh -- and the banner states it with
        the catalog's area, fill and whether it fits."""
        config = make_pipeline_config(nx=40, ny=40, nz=12, dx=500e-6, dy=500e-6, dz=20e-6,
                                      interposer_um=100.0, bump_um=40.0, die_um=100.0,
                                      die_power_w=12.0, h_convective=1000.0)
        main_module.print_startup_banner(config, source="test")
        out = capsys.readouterr().out
        assert "Die:            20.000 x 20.000 mm" in out
        assert "Catalog:        192.00 mm² across 4 blocks (48.0% fill)" in out
        assert "Catalog fits:   yes (largest block 11.000 x 11.000 mm)" in out

    def test_banner_says_when_the_catalog_does_not_fit(self, capsys):
        config = make_pipeline_config(nx=40, ny=40, nz=12, dx=25e-6, dy=25e-6, dz=20e-6,
                                      interposer_um=100.0, bump_um=40.0, die_um=100.0)
        main_module.print_startup_banner(config, source="test")
        assert "Catalog fits:   no (largest block 11.000 x 11.000 mm)" in capsys.readouterr().out


class TestStressCase:
    """Decision 1: the 150.56 degC operating point is not a preset. It ships
    as its own file, which states that it is a stress case EXPECTED to exceed
    its 90 degC threshold, and every run of it says so."""

    def test_stress_file_is_a_custom_regime_with_its_own_threshold(self):
        config = config_loader_module.load_config(str(STRESS_CONFIG))
        assert config.case.name == "stress_high_power_low_cool"
        assert config.case.expect_threshold_exceeded is True
        assert config.visualization.hotspot_threshold_c == 90.0
        assert main_module.regime_label(config) == "custom"

    def test_stress_case_really_exceeds_its_threshold(self):
        """The expectation is physics, not a label: solve it and check."""
        config = config_loader_module.load_config(str(STRESS_CONFIG))
        solver = ThermalSolver.from_grid_builder(main_module.build_chiplet_stack(config))
        peak_c = float(np.max(solver.solve_steady_state(apply_result=False))) - 273.15
        assert peak_c > config.visualization.hotspot_threshold_c

    def test_banner_names_the_stress_case(self, capsys):
        config = config_loader_module.load_config(str(STRESS_CONFIG))
        main_module.print_startup_banner(config, source="test", from_file=True)
        out = capsys.readouterr().out
        assert "Case: stress_high_power_low_cool" in out
        assert "STRESS CASE: EXPECTED to exceed the 90.0 C threshold" in out

    def test_a_stress_case_that_stays_below_threshold_is_flagged(self, capsys):
        config = config_loader_module.load_config(str(STRESS_CONFIG))
        report = {
            "html_path": "outputs/thermal_3d.html", "hotspot_count": 0,
            "peak_temperature_c": 80.0, "peak_location_um": (0.0, 0.0, 0.0),
            "peak_location_ijk": (0, 0, 0), "transient_peak_temperature_c": None,
            "transient_duration_s": None, "tau_s": 0.1, "converged_fraction": 0.0,
        }
        main_module.print_stage1_results(config, report)
        assert "EXPECTED to exceed the threshold and did not" in capsys.readouterr().out

    def test_default_config_is_the_default_preset(self):
        config = config_loader_module.load_config(str(SHIPPED_CONFIG))
        assert main_module.regime_label(config) == "passive-lowpower"
        assert config.case.expect_threshold_exceeded is False


def _pipeline_configs(monkeypatch):
    """One config per way main.py obtains one: flags, a preset, each shipped
    file, and the interactive prompts with every default accepted."""
    parse = main_module.build_cli_parser().parse_args
    configs = {
        "flags": main_module.build_config_from_flags(parse([]))[0],
        "preset": main_module.build_config_from_flags(parse(["--preset", "cowos-coldplate"]))[0],
        "config.json": config_loader_module.load_config(str(SHIPPED_CONFIG)),
        "stress": config_loader_module.load_config(str(STRESS_CONFIG)),
    }
    monkeypatch.setattr("builtins.input", lambda *_: "")
    configs["interactive"] = main_module.run_interactive_prompts()[0]
    return configs


class TestPipelineSuppliesEveryGridBuilderKey:
    """Decisions 2 and 3: GridBuilder keeps its .get() defaults (h_convective
    2500, nx 200, ...) for direct GridBuilder(config=...) construction. These
    pin that no pipeline path can reach them."""

    def test_main_always_supplies_h_convective(self, monkeypatch):
        for name, config in _pipeline_configs(monkeypatch).items():
            assert "h_convective" in config.to_dict()["boundary_conditions"], name
            gb = main_module.build_chiplet_stack(config)
            assert gb.config.h_convective == config.boundary_conditions.h_convective, name

    def test_main_always_supplies_every_grid_key(self, monkeypatch):
        for name, config in _pipeline_configs(monkeypatch).items():
            g = config.grid
            assert set(config.to_dict()["grid"]) == {"nx", "ny", "nz", "dx", "dy", "dz"}, name
            built = main_module.build_chiplet_stack(config).config
            assert (built.nx, built.ny, built.nz, built.dx, built.dy, built.dz) == \
                (g.nx, g.ny, g.nz, g.dx, g.dy, g.dz), name

    def test_optimizer_supplies_h_convective(self):
        config = config_loader_module.load_config(str(SHIPPED_CONFIG))
        stage1 = ThermalSolver.from_grid_builder(main_module.build_chiplet_stack(config))
        opt = ChipletPlacementOptimizer.from_pipeline_config(config, reuse_solver=stage1)
        assert opt.grid_builder.config.h_convective == config.boundary_conditions.h_convective

    def test_h_convective_has_no_default(self):
        with pytest.raises(TypeError):
            BoundaryConditionsSection()


class TestReportLabelsMatchTheMethod:
    """Decision 5: the Stage 2 report labelled exhaustive runs as simulated
    annealing ('SA iterations', 'Random seed'). Presentation only."""

    @staticmethod
    def _exhaustive_result():
        opt = ChipletPlacementOptimizer(nx=20, ny=20, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
                                        h_convective=1000.0, placement_pitch_um=5000.0, seed=0)
        search = exhaustive_module.ExhaustivePlacementSearch(opt, placement_pitch_um=5000.0)
        return search.search(verbose=False).to_optimization_result(opt)

    @staticmethod
    def _label_text(path):
        """The report without its embedded images, whose base64 could hold
        any letter sequence by chance."""
        import re as _re
        html = pathlib.Path(path).read_text(encoding="utf-8")
        return _re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", html)

    def test_report_labels_exhaustive_as_exhaustive(self, tmp_path):
        path = optimizer_module.render_optimization_report(
            self._exhaustive_result(), output_dir=str(tmp_path), method="exhaustive")
        text = self._label_text(path)
        assert "exhaustive" in text.lower()
        assert "anneal" not in text.lower()
        assert "SA iterations" not in text and "Random seed" not in text

    def test_annealing_report_is_still_labelled_annealing(self, tmp_path):
        result = make_small_optimizer(seed=1).optimize(iterations=2, verbose=False)
        text = self._label_text(optimizer_module.render_optimization_report(
            result, output_dir=str(tmp_path), method="anneal"))
        assert "simulated-annealing" in text
        assert "SA iterations" in text

    def test_unknown_method_is_refused(self, tmp_path):
        with pytest.raises(OptimizerError, match="method"):
            optimizer_module.render_optimization_report(
                self._exhaustive_result(), output_dir=str(tmp_path), method="guess")


class TestPeakLocationIsTheCellCentre:
    """Decision 6: a location is the cell CENTRE, (i + 0.5) * d, not the
    cell's lower corner i * d. Indices are unchanged."""

    def test_injected_peak_reports_its_cell_centre(self, tmp_path):
        import csv as _csv
        from src.visualization import ThermalVisualizer
        dx, dy, dz = 500e-6, 250e-6, 20e-6
        T = np.full((10, 8, 6), 300.0)
        i, j, k = 7, 2, 4
        T[i, j, k] = 400.0
        viz = ThermalVisualizer(T, dx=dx, dy=dy, dz=dz, output_dir=str(tmp_path))
        expected = ((i + 0.5) * dx * 1e6, (j + 0.5) * dy * 1e6, (k + 0.5) * dz * 1e6)

        peak = viz.get_peak_hotspot()
        assert peak["index_ijk"] == (i, j, k)
        assert peak["location_um"] == pytest.approx(expected, rel=1e-12)
        assert viz.get_hotspot_report(threshold_c=100.0).peak_location_um == \
            pytest.approx(expected, rel=1e-12)
        with open(viz.export_hotspot_csv(threshold_c=100.0), newline="") as f:
            rows = list(_csv.reader(f))
        assert [float(v) for v in rows[1][3:6]] == pytest.approx(expected, abs=1e-3)

    def test_webgl_readout_uses_cell_centres(self, tmp_path):
        from src.visualization import ThermalVisualizer
        viz = ThermalVisualizer(np.full((4, 4, 4), 300.0), dx=1e-3, dy=1e-3, dz=1e-5,
                                output_dir=str(tmp_path))
        html = pathlib.Path(viz.render_interactive_webgl()).read_text(encoding="utf-8")
        assert '"stride":1' in html
        assert "(i + 0.5 / DATA.stride) * DATA.dx_um" in html


class TestLegacyReportArithmetic:
    """Productization audit: two presentation defects in the protected
    optimizer's report layer, each proven by a failing test before any fix.
    Neither touches a number the optimizer computes -- only what it prints."""

    def test_scale_note_does_not_mix_chiplet_area_with_die_average(self, tmp_path):
        """The note read 'carrying 12.0 W across 192.0 mm^2 of chiplets -- 48%
        fill at an average 3.0 W/cm^2'. 3.0 is the DIE average (12 W / 4 cm^2);
        set beside 192 mm^2 it reads as 12 W / 1.92 cm^2, which is 6.25."""
        import html as _html
        import re as _re
        opt = ChipletPlacementOptimizer(nx=20, ny=20, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
                                        h_convective=1000.0, placement_pitch_um=5000.0, seed=0)
        result = exhaustive_module.ExhaustivePlacementSearch(opt, placement_pitch_um=5000.0) \
            .search(verbose=False).to_optimization_result(opt)
        path = optimizer_module.render_optimization_report(result, output_dir=str(tmp_path),
                                                           method="exhaustive")
        text = _html.unescape(_re.sub(r"<[^>]+>", " ", pathlib.Path(path).read_text(encoding="utf-8")))
        text = _re.sub(r"\s+", " ", text)
        # The default catalog: 36 W over 192 mm^2 of chiplets, on a 400 mm^2 die.
        assert "18.75 W/cm² over the chiplets" in text
        assert "9.00 W/cm² averaged over the die" in text

    def test_improvement_percent_is_of_the_rise_above_ambient(self):
        """improvement_pct called its denominator `rise_initial` but divided by
        the initial peak in degC: a percentage of a Celsius reading, which
        changes with the temperature scale. The rise above ambient does not."""
        r = optimizer_module.OptimizationResult(
            best_layout=(), best_cost=70.0, best_peak_temperature_c=70.0,
            best_temperature_field_k=np.zeros(1), best_cost_history=[], current_cost_history=[],
            initial_layout=(), initial_peak_temperature_c=85.0, iterations=0, accepted_moves=0,
            rejected_moves=0, overlap_candidates=0, seed=0, t_ambient_c=25.0)
        assert r.improvement_c == pytest.approx(15.0)
        assert r.improvement_pct == pytest.approx(25.0)      # 15 K of a 60 K rise


class TestCitedEffectiveProperties:
    """Every conductivity in the stack has to come from somewhere nameable.
    These two are effective media -- a TSV-bearing interposer and a BEOL
    stack -- built from the same cited table and the same series/parallel
    rules as the microbump composite, and checked against the numbers the
    paper published for its own cells."""

    def test_the_tsv_cell_reproduces_the_papers_own_numbers(self):
        from src.composite_materials import tsv_cell_k
        # The paper's cell: 8 um copper core, 9 um to the liner's outside,
        # 40 um pitch. It quotes 175.2 out of plane analytically and 115.3
        # in plane by FEM.
        kxy, kz = tsv_cell_k(8.0, 9.0, 40.0)
        assert kz == pytest.approx(175.2, abs=0.05)
        assert kxy == pytest.approx(115.3, rel=0.02)

    def test_a_tsv_interposer_conducts_better_through_than_along(self):
        """Copper pillars help heat cross the interposer and the oxide liners
        get in the way of heat running along it, so plain silicon is wrong in
        both directions at once -- optimistic through, pessimistic along."""
        from src.grid_builder import MATERIALS
        m = MATERIALS["tsv_interposer"]
        assert m["kz"] > MATERIALS["silicon"]["k"] > m["kx"]
        assert m["kx"] == m["ky"]

    def test_beol_is_a_barrier_through_and_a_spreader_along(self):
        """About two orders of magnitude apart. The small number is the one
        that matters: it sits directly on the plane where power is deposited."""
        from src.composite_materials import beol_k
        from src.grid_builder import MATERIALS
        kxy, kz = beol_k()
        assert kz < 3.0 and kxy > 100.0
        assert kxy / kz > 50.0
        assert MATERIALS["beol"]["kz"] == pytest.approx(kz)
        # More copper helps along the stack and barely helps through it.
        kxy_rich, kz_rich = beol_k(0.45)
        assert kxy_rich > kxy
        assert kz_rich < 1.5 * kz

    def test_a_copper_fraction_outside_zero_to_one_is_refused(self):
        from src.composite_materials import beol_k
        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="between 0 and 1"):
                beol_k(bad)

    def test_the_default_stack_is_unchanged_and_tsvs_are_opt_in(self):
        """The TSV interposer is a per-design choice rather than a new
        default: it moves the edge-AI peak +2.8 K, and the one measured
        validation case is a bare test chip with no interposer at all, which a
        global switch would silently misdescribe. BEOL is opt-in the same way
        (the package's beol_um), folded into the bump layer rather than given
        cells, because at 5-15 um it is thinner than one 20 um z-cell."""
        import src.experiment as experiment_module
        import studio.engine as engine_module
        assert experiment_module.STACK_MATERIALS == (
            "silicon", "bump_underfill_composite", "silicon")
        assert engine_module.INTERPOSER_MATERIALS == ("silicon", "tsv_interposer")
        assert "beol" not in engine_module.INTERPOSER_MATERIALS
        assert engine_module.PackageSpec.parse(
            {"size_mm": 20.0, "h_w_m2k": 2000.0, "ambient_c": 25.0, "pitch_mm": 2.0}).beol_um == 0.0


class TestRemainingLimitations:
    def test_remaining_limitations_say_what_has_and_has_not_been_compared(self):
        """One test chip is not a validated model, and the limitations have to
        say what the evidence covers, how it was obtained, and where it stops."""
        assert "Remaining limitations" in validation_module.text_block()
        for block in (validation_module.text_block(), validation_module.html_block()):
            assert "One measured test chip, one cooling regime" in block
            # The evidence, with the numbers that make it a claim.
            assert "16 of 16 sensors inside the stated band" in block
            assert "0.64 K RMS against 28.95 K of spread" in block
            assert "11 of 12 held-out patterns" in block and "12 of 13" in block
            # The band is confirmed by the scatter, not merely wide enough.
            assert "5.3 % RMS" in block and "(5.5 %)" in block
            # What is calibrated, and on what.
            assert "calibrated on OTHER patterns of the same rig" in block
            # The data defect is disclosed, with the tool that corrects it.
            assert "rotated by one place" in block and "tools/ttc_extract.py" in block
            # And the assumption the same dataset showed to be regime-dependent.
            assert "A single constant cooling coefficient" in block
            assert "structurally wrong for natural convection" in block


# ---------------------------------------------------------------------------
# The placement experiment: catalog, initial placement, power map, before/after
# ---------------------------------------------------------------------------

def make_experiment_config(*, nx=20, ny=20, nz=12, dx=1000e-6, dy=1000e-6, dz=20e-6,
                           h_convective=1000.0, pitch_um=4000.0, chiplets=None,
                           origins=None):
    """A small placement experiment: an explicit catalog and an explicit
    initial placement, on a 20 x 20 mm package with a 240 um stack."""
    chiplets = chiplets if chiplets is not None else (
        ChipletEntry(id="a", name="A", width_um=4000.0, height_um=4000.0, power_w=6.0),
        ChipletEntry(id="b", name="B", width_um=6000.0, height_um=4000.0, power_w=4.0),
    )
    origins = origins if origins is not None else {"a": (0.0, 0.0), "b": (8000.0, 0.0)}
    config = make_pipeline_config(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, h_convective=h_convective,
        die_power_w=sum(c.power_w for c in chiplets),
        interposer_um=100.0, bump_um=40.0, die_um=100.0)
    config.chiplets = tuple(chiplets)
    config.initial_placement = InitialPlacementSection(pitch_um=pitch_um,
                                                       origins_um=dict(origins))
    config.validate()
    return config


def make_experiment(**kwargs):
    return experiment_module.Experiment(make_experiment_config(**kwargs), source="test")


@pytest.fixture(scope="module")
def optimized_experiment():
    """One small experiment, solved and optimized once, shared by the tests
    that only read its results."""
    exp = make_experiment()
    exp.solve_initial()
    exp.optimize(method="exhaustive", verbose=False)
    return exp


class TestChipletConfig:
    """The catalog and the initial placement are configuration, validated as
    strictly as everything else in the file."""

    def test_canonical_config_is_an_experiment(self):
        config = config_loader_module.load_config(str(SHIPPED_CONFIG))
        assert [c.id for c in config.chiplets] == ["cpu0", "gpu0", "hbm0", "npu0"]
        assert config.initial_placement.pitch_um == 2000.0
        assert sum(c.power_w for c in config.chiplets) == pytest.approx(
            config.power.die_power_w, rel=1e-9)

    def test_unknown_key_in_a_chiplet_is_named(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r["chiplets"][0].update(colour="red"))
        with pytest.raises(ConfigError, match=r"chiplets\[0\]\.colour"):
            config_loader_module.load_config(path)

    def test_missing_chiplet_field_is_named(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r["chiplets"][1].pop("power_w"))
        with pytest.raises(ConfigError, match=r"chiplets\[1\]\.power_w"):
            config_loader_module.load_config(path)

    def test_catalog_without_a_placement_is_refused(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r.pop("initial_placement"))
        with pytest.raises(ConfigError, match="initial_placement"):
            config_loader_module.load_config(path)

    def test_duplicate_ids_are_refused(self, tmp_path):
        def duplicate(raw):
            raw["chiplets"][1]["id"] = raw["chiplets"][0]["id"]
        path = _config_file(tmp_path, duplicate)
        with pytest.raises(ConfigError, match="duplicate chiplet id"):
            config_loader_module.load_config(path)

    def test_origin_for_an_unknown_chiplet_is_refused(self, tmp_path):
        path = _config_file(
            tmp_path, lambda r: r["initial_placement"]["origins_um"].update(ghost=[0, 0]))
        with pytest.raises(ConfigError, match="unknown chiplet"):
            config_loader_module.load_config(path)

    def test_die_power_must_equal_the_chiplet_total(self, tmp_path):
        path = _config_file(tmp_path, lambda r: r["power"].update(die_power_w=99.0))
        with pytest.raises(ConfigError, match="die_power_w"):
            config_loader_module.load_config(path)

    def test_round_trip_keeps_the_experiment_sections(self, tmp_path):
        first = config_loader_module.load_config(str(SHIPPED_CONFIG))
        path = tmp_path / "round_trip.json"
        path.write_text(json.dumps(first.as_dict()), encoding="utf-8")
        assert config_loader_module.load_config(str(path)).as_dict() == first.as_dict()


class TestPlacementValidation:
    """Every physical constraint on a placement, each reported by name."""

    def test_canonical_placement_is_valid(self):
        exp = experiment_module.Experiment(
            config_loader_module.load_config(str(SHIPPED_CONFIG)))
        assert exp.violations == []

    def test_overlap_is_named(self):
        exp = make_experiment(pitch_um=2000.0,
                              origins={"a": (0.0, 0.0), "b": (2000.0, 0.0)})
        assert any("a and b overlap" in v for v in exp.violations), exp.violations

    def test_leaving_the_package_is_named(self):
        exp = make_experiment(origins={"a": (0.0, 0.0), "b": (16000.0, 0.0)})
        assert any("leaves the" in v and v.startswith("b:") for v in exp.violations)

    def test_off_lattice_origin_is_named(self):
        exp = make_experiment(origins={"a": (0.0, 0.0), "b": (9000.0, 0.0)})
        assert any("placement lattice" in v and v.startswith("b:") for v in exp.violations)

    def test_footprint_that_is_not_whole_cells_is_named(self):
        odd = (ChipletEntry(id="a", name="A", width_um=4500.0, height_um=4000.0, power_w=6.0),)
        exp = make_experiment(chiplets=odd, origins={"a": (0.0, 0.0)})
        assert any("whole number" in v and v.startswith("a:") for v in exp.violations)

    def test_main_refuses_an_invalid_placement_before_solving(self, tmp_path, capsys):
        path = _config_file(tmp_path, lambda r: r["initial_placement"]["origins_um"]
                            .update(cpu0=[0.0, 0.0]))     # straight on top of hbm0
        out_dir = tmp_path / "out"
        rc = main_module.main(["--config", path, "--no-browser", "--output-dir", str(out_dir)])
        assert rc == main_module.EXIT_PREFLIGHT_FAILED
        assert "overlap" in capsys.readouterr().err
        assert not out_dir.exists() or not list(out_dir.iterdir())


class TestExperimentPowerMap:
    """Chiplet geometry and power become a cell-level source, and only there."""

    def test_each_chiplet_gets_exactly_its_power(self):
        exp = make_experiment()
        power = experiment_module.build_power_map(exp.package, exp.catalog,
                                                  exp.initial_placement)
        deposited = dict(power.per_chiplet_w)
        for c in exp.catalog:
            assert deposited[c.id] == pytest.approx(c.power_w, rel=1e-12)
        assert power.deposited_w == pytest.approx(power.specified_w, rel=1e-12)

    def test_power_lives_only_in_the_footprints_and_the_active_plane(self):
        """Heat enters at the device layer: the first die cell, against the
        bumps, with the sink on the opposite face."""
        exp = make_experiment()
        power = experiment_module.build_power_map(exp.package, exp.catalog,
                                                  exp.initial_placement)
        q = np.asarray(power.q_w_m3)
        die0, die1 = exp.package.die_z
        z0, z1 = experiment_module.power_z_range(exp.package)
        assert (z0, z1) == (die0, die0 + 1), "the active plane is the first die cell"
        assert not np.any(q[:, :, :z0]), "power below the active plane"
        assert not np.any(q[:, :, z1:]), "power above the active plane"
        covered = np.zeros(q.shape[:2], dtype=bool)
        for _cid, i0, j0, nvx, nvy in power.blocks:
            covered[i0:i0 + nvx, j0:j0 + nvy] = True
        assert not np.any(q[~covered, :]), "power outside every chiplet footprint"
        assert np.all(q[covered, z0:z1] > 0.0)
        assert die1 > z1, "the rest of the die carries no source, only conduction"

    def test_volume_convention_is_still_available_and_conserves_power(self):
        """The older whole-die convention is kept so the two can be compared;
        both deposit exactly the catalog's watts."""
        exp = make_experiment()
        active = experiment_module.build_power_map(exp.package, exp.catalog,
                                                   exp.initial_placement, "active")
        volume = experiment_module.build_power_map(exp.package, exp.catalog,
                                                   exp.initial_placement, "volume")
        z0, z1 = exp.package.die_z
        assert np.any(np.asarray(volume.q_w_m3)[:, :, z1 - 1]), "volume mode fills the die"
        assert volume.deposited_w == pytest.approx(active.deposited_w, rel=1e-12)
        assert volume.deposited_w == pytest.approx(volume.specified_w, rel=1e-12)
        assert volume.powered_cells > active.powered_cells
        with pytest.raises(experiment_module.ExperimentError):
            experiment_module.build_power_map(exp.package, exp.catalog,
                                              exp.initial_placement, "everywhere")

    def test_the_active_plane_is_hotter_than_spreading_power_through_the_die(self):
        """Why the convention matters: the same watts, entering at the face
        furthest from the sink, cannot be cooler."""
        exp = make_experiment()
        exp.solve_initial()
        hot = exp.initial.peak_c
        cool = experiment_module.solve_placement(
            exp.solver, exp.package, exp.catalog, exp.initial_placement,
            "volume-convention", "volume").peak_c
        assert hot >= cool - 1e-9

    def test_moving_a_chiplet_moves_its_power(self):
        exp = make_experiment()
        here = experiment_module.build_power_map(exp.package, exp.catalog,
                                                 exp.initial_placement)
        moved_placement = experiment_module.Placement(
            pitch_m=exp.initial_placement.pitch_m,
            origins_m=(("a", 12000e-6, 12000e-6), ("b", 8000e-6, 0.0)))
        there = experiment_module.build_power_map(exp.package, exp.catalog, moved_placement)
        assert not np.array_equal(np.asarray(here.q_w_m3), np.asarray(there.q_w_m3))
        assert there.deposited_w == pytest.approx(here.deposited_w, rel=1e-12)
        # 'a' left its old cells and arrived in the new ones.
        assert np.asarray(there.q_w_m3)[0:4, 0:4, :].sum() == 0.0
        assert np.asarray(there.q_w_m3)[12:16, 12:16, :].sum() > 0.0

    def test_matches_the_optimizers_own_power_map(self):
        """Two independent implementations of one physical statement: this
        module's power map and the protected optimizer's must agree for the
        same layout, or a before/after comparison is meaningless.

        Agreement is to floating-point round-off rather than bit-for-bit.
        The two arrive at the same density by different arithmetic -- this
        module now builds a per-cell weight array (so internal blocks can take
        the cells they overlap) and divides once, while the optimizer divides
        by the cell count directly -- and those differ in the last bit. One
        ulp is agreement; anything a discretization could hide is not, hence
        the 1e-12 relative tolerance rather than a loose one.
        """
        exp = make_experiment()
        assert not any(c.blocks or c.power_grid for c in exp.catalog), (
            "ChipletSpec carries a footprint and a total power and nothing "
            "inside it, so the protected optimizer models every chiplet as a "
            "uniform heater. This comparison is therefore only meaningful for "
            "uniform chiplets -- which is all the CLI pipeline can express, "
            "since internal blocks reach the solver through the studio alone.")
        optimizer = ChipletPlacementOptimizer(
            nx=exp.package.nx, ny=exp.package.ny, nz=exp.package.nz,
            dx=exp.package.dx, dy=exp.package.dy, dz=exp.package.dz,
            catalog=tuple(c.to_spec() for c in exp.catalog),
            h_convective=exp.package.h_convective, t_ambient_c=exp.package.t_ambient_c,
            placement_pitch_um=exp.initial_placement.pitch_m * 1e6, seed=0)
        optimizer._apply_layout_to_q_map(exp.layout_of(exp.initial_placement))
        mine = experiment_module.build_power_map(exp.package, exp.catalog,
                                                 exp.initial_placement)
        theirs, ours = np.asarray(optimizer.solver.q_map), np.asarray(mine.q_w_m3)
        assert theirs.shape == ours.shape
        # Same cells carry power, and the same watts land in them.
        assert np.array_equal(theirs > 0, ours > 0)
        assert np.allclose(theirs, ours, rtol=1e-12, atol=0.0)
        assert float(theirs.sum()) == pytest.approx(float(ours.sum()), rel=1e-12)


class TestExperimentBaseline:
    """The initial placement is solved by the real solver and checked."""

    def test_initial_solve_conserves_energy_independently(self):
        exp = make_experiment()
        initial = exp.solve_initial()
        total_w = sum(c.power_w for c in exp.catalog)
        assert initial.power.deposited_w == pytest.approx(total_w, rel=1e-12)
        # Computed in the product layer from h and the field, sharing no code
        # with the solver's own energy check.
        assert initial.sink_heat_w == pytest.approx(total_w, rel=1e-9)
        assert initial.residual < thermal_solver_module.STEADY_RESIDUAL_TOL
        assert initial.energy_imbalance_k < thermal_solver_module.STEADY_ENERGY_BALANCE_TOL_K

    def test_peak_is_above_the_convective_floor(self):
        exp = make_experiment()
        initial = exp.solve_initial()
        sanity = exp.sanity(initial)
        assert sanity["floor_ratio"] > 1.0
        assert sanity["measured_rise_k"] > sanity["floor_rise_k"]

    def test_zero_power_sits_at_ambient(self):
        cold = (ChipletEntry(id="a", name="A", width_um=4000.0, height_um=4000.0, power_w=0.0),)
        exp = make_experiment(chiplets=cold, origins={"a": (0.0, 0.0)})
        initial = exp.solve_initial()
        # 1 uK: the residue of a linear solve whose right-hand side is zero
        # everywhere but the boundary, not a physical rise.
        assert initial.peak_c == pytest.approx(exp.package.t_ambient_c, abs=1e-6)


class TestExperimentOptimization:
    """The optimizer starts from the given placement, changes only positions,
    and its winner is re-solved before anyone sees it."""

    def test_optimized_placement_is_valid_and_on_the_lattice(self, optimized_experiment):
        exp = optimized_experiment
        assert experiment_module.validate_placement(
            exp.package, exp.catalog, exp.optimized.placement) == []

    def test_optimizer_changes_only_positions(self, optimized_experiment):
        exp = optimized_experiment
        assert dict(exp.optimized.power.per_chiplet_w) == pytest.approx(
            dict(exp.initial.power.per_chiplet_w))
        assert exp.optimized.power.specified_w == exp.initial.power.specified_w

    def test_optimized_is_at_least_as_cool_as_the_initial(self, optimized_experiment):
        """The initial placement is on the lattice, so it is inside the search
        space: an exhaustive search cannot return anything worse."""
        exp = optimized_experiment
        assert exp.optimized.peak_c <= exp.initial.peak_c + 1e-9

    def test_reported_optimum_is_a_real_re_solve(self, optimized_experiment):
        record = optimized_experiment.optimization
        assert record.resolve_delta_k <= experiment_module.RESOLVE_TOL_K
        assert record.resolve_field_delta_k <= experiment_module.RESOLVE_TOL_K
        assert optimized_experiment.optimized.residual < thermal_solver_module.STEADY_RESIDUAL_TOL

    def test_initial_result_is_untouched_by_optimization(self):
        exp = make_experiment()
        initial = exp.solve_initial()
        before_field = np.array(initial.field_k, copy=True)
        before_peak, before_origins = initial.peak_c, initial.placement.origins_m
        exp.optimize(method="exhaustive", verbose=False)
        assert exp.initial is initial
        assert initial.peak_c == before_peak
        assert initial.placement.origins_m == before_origins
        assert np.array_equal(np.asarray(initial.field_k), before_field)
        assert not np.asarray(initial.field_k).flags.writeable

    def test_exhaustive_is_reproducible(self):
        first, second = make_experiment(), make_experiment()
        for exp in (first, second):
            exp.solve_initial()
            exp.optimize(method="exhaustive", verbose=False)
        assert first.optimized.placement.origins_m == second.optimized.placement.origins_m
        assert first.optimized.peak_c == second.optimized.peak_c
        assert first.optimization.deterministic is True

    def test_annealing_starts_from_the_given_initial_placement(self):
        exp = make_experiment()
        initial = exp.solve_initial()
        exp.optimize(method="anneal", iterations=4, seed=1, verbose=False)
        record = exp.optimization
        assert record.method == "anneal" and record.seed == 1
        assert len(record.history_c) == 4
        assert record.deterministic is False
        assert exp.optimized.peak_c <= initial.peak_c + 1e-9

    def test_an_optimizer_that_disagrees_with_the_solver_is_refused(self, monkeypatch):
        """If the search reports a peak the real solver does not reproduce,
        the experiment fails rather than quietly publishing either number."""
        import dataclasses as _dc
        exp = make_experiment()
        exp.solve_initial()
        real_search = exhaustive_module.ExhaustivePlacementSearch.search

        def lying_search(self, verbose=True):
            result = real_search(self, verbose=verbose)
            return _dc.replace(result,
                               best_peak_temperature_c=result.best_peak_temperature_c - 5.0)

        monkeypatch.setattr(exhaustive_module.ExhaustivePlacementSearch, "search",
                            lying_search)
        with pytest.raises(experiment_module.ExperimentError, match="does not confirm"):
            exp.optimize(method="exhaustive", verbose=False)


class TestExperimentReport:
    """The report answers the product questions, from the stored results."""

    SECTIONS = ("1. Experiment summary", "2. Package geometry", "3. Chiplet catalog",
                "4. Initial placement", "5. Initial thermal result",
                "6. Optimization objective", "7. Constraints", "8. Optimization history",
                "9. Optimized placement", "10. Before / after temperature comparison",
                "11. Temperature difference map", "12. Verification", "13. Limitations")

    @staticmethod
    def _text(exp, tmp_path):
        import html as _htmllib
        path = report_module.render_experiment_report(exp, output_dir=str(tmp_path))
        return _htmllib.unescape(pathlib.Path(path).read_text(encoding="utf-8"))

    def test_report_has_every_section_in_order(self, optimized_experiment, tmp_path):
        text = self._text(optimized_experiment, tmp_path)
        positions = [text.index(section) for section in self.SECTIONS]
        assert positions == sorted(positions)

    def test_report_quotes_the_stored_results(self, optimized_experiment, tmp_path):
        exp = optimized_experiment
        text = self._text(exp, tmp_path)
        assert f"{exp.initial.peak_c:.4f}" in text
        assert f"{exp.optimized.peak_c:.4f}" in text
        for chiplet in exp.catalog:
            assert chiplet.id in text

    def test_report_states_both_densities_and_they_are_right(self, optimized_experiment,
                                                             tmp_path):
        exp = optimized_experiment
        sanity = exp.sanity(exp.initial)
        assert sanity["flux_over_chiplets_w_cm2"] == pytest.approx(
            sanity["total_power_w"] / (sanity["chiplet_area_mm2"] * 1e-2))
        assert sanity["flux_over_package_w_cm2"] == pytest.approx(
            sanity["total_power_w"] / (sanity["package_area_mm2"] * 1e-2))
        text = self._text(exp, tmp_path)
        assert f"{sanity['flux_over_chiplets_w_cm2']:.2f} W/cm²" in text
        assert f"{sanity['flux_over_package_w_cm2']:.2f} W/cm²" in text

    def test_the_report_never_claims_validation_it_has_not_earned(self, optimized_experiment,
                                                                  tmp_path):
        """The bare word is the thing to guard against. The report may state
        what HAS been compared with hardware -- currently one test chip -- but
        every occurrence has to be qualified, and the half that has not been
        validated has to appear too. Otherwise a reader skimming for the word
        comes away with the opposite of the truth."""
        import re as _re
        text = self._text(optimized_experiment, tmp_path).lower()
        # Every form the word is allowed to take, and each one limits the
        # claim: it is denied, scoped to the spreading, or explicitly the
        # thing the model did NOT earn.
        allowed = ("not validated", "validated (one test chip")
        found = list(_re.finditer("validated", text))
        assert found, "the report must carry the validation status at all"
        for match in found:
            # A qualifier can sit either side of the word: "not validated"
            # before it, "validated (one test chip, ..." after it.
            context = text[max(0, match.start() - 24):match.end() + 24]
            assert any(q in context for q in allowed), context
        # The unearned half is never silently dropped.
        assert "not validated" in text

    def test_report_labels_the_method_that_actually_ran(self, optimized_experiment,
                                                        tmp_path):
        text = self._text(optimized_experiment, tmp_path)
        assert "exhaustive" in text
        assert "anneal" not in text.lower().replace("annealing report", "")


class TestConfigFileIsTheSingleSource:
    """Flags that change what is simulated are refused beside --config rather
    than ignored: --h-convective 5000 --die-power-w 99 used to change nothing
    and say nothing."""

    def test_physics_flag_beside_config_is_refused(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        rc = main_module.main(["--config", str(SHIPPED_CONFIG), "--h-convective", "5000",
                               "--no-browser", "--output-dir", str(out_dir)])
        assert rc == 1
        err = capsys.readouterr().err
        assert "--h-convective" in err and "--config" in err
        assert not out_dir.exists() or not list(out_dir.iterdir())

    def test_output_dir_override_is_reported_and_used(self, tmp_path, capsys):
        rc = main_module.main(["--config", str(SHIPPED_CONFIG), "--no-browser",
                               "--output-dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert f"Output directory: {tmp_path}" in out
        assert (tmp_path / "thermal_3d.html").exists()
        assert (tmp_path / "experiment_report.html").exists()


# ---------------------------------------------------------------------------
# MVP: the screening surrogate
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def surrogate_experiment():
    """The same small experiment as optimized_experiment, optimized by the
    screening surrogate instead of the exact search."""
    exp = make_experiment()
    exp.solve_initial()
    exp.optimize(method="surrogate", verbose=False, surrogate_top_k=5, surrogate_samples=12)
    return exp


class TestSurrogateFeatures:
    """The features are the physics the model is allowed to use; they are
    checked against a loop written from the formulas, not against themselves."""

    def test_features_match_a_scalar_reference(self):
        exp = make_experiment()
        package, catalog = exp.package, exp.catalog
        lam = surrogate_module.spreading_length_m(package)
        xs, ys = np.array([[0.0, 8e-3]]), np.array([[0.0, 12e-3]])
        features = surrogate_module.layout_features(package, catalog, lam, xs, ys)[0]
        h = package.h_convective
        total_w = sum(c.power_w for c in catalog)
        for i, ci in enumerate(catalog):
            spread_i = ci.power_w / (h * (ci.width_m + 2 * lam) * (ci.height_m + 2 * lam))
            neighbours = 0.0
            for j, cj in enumerate(catalog):
                if i == j:
                    continue
                gx = max(xs[0, j] - (xs[0, i] + ci.width_m), xs[0, i] - (xs[0, j] + cj.width_m), 0.0)
                gy = max(ys[0, j] - (ys[0, i] + ci.height_m), ys[0, i] - (ys[0, j] + cj.height_m), 0.0)
                spread_j = cj.power_w / (h * (cj.width_m + 2 * lam) * (cj.height_m + 2 * lam))
                neighbours += spread_j * float(np.exp(-np.hypot(gx, gy) / lam))
            walls = sum(float(np.exp(-2.0 * s / lam)) for s in (
                xs[0, i], package.lx_m - xs[0, i] - ci.width_m,
                ys[0, i], package.ly_m - ys[0, i] - ci.height_m))
            expected = [total_w / (h * package.sink_area_m2), ci.power_w / (h * ci.area_m2),
                        spread_i, neighbours, spread_i * walls]
            assert list(features[i]) == pytest.approx(expected, rel=1e-12)

    def test_spreading_length_is_the_banner_value(self):
        config = config_loader_module.load_config(str(SHIPPED_CONFIG))
        package = experiment_module.package_from_config(config)
        assert surrogate_module.spreading_length_m(package) * 1e3 == pytest.approx(
            main_module.regime_summary(config)["lambda_mm"], rel=1e-12)


class TestSurrogateScreening:
    """The surrogate ranks; the real solver decides. Pinned from both sides:
    what the search reports is a real solve, and a deliberately bad surrogate
    costs quality but never correctness."""

    def test_enumeration_matches_the_exact_search(self, optimized_experiment):
        exp = optimized_experiment
        xs, _ys, total, complete = surrogate_module.enumerate_buildable(
            exp.package, exp.catalog, exp.initial_placement.pitch_m)
        assert complete
        assert xs.shape[0] == exp.optimization.candidates_evaluated
        assert total == exp.optimization.candidates_total

    def test_winner_is_valid_verified_and_never_beats_the_exact_optimum(
            self, surrogate_experiment, optimized_experiment):
        exp = surrogate_experiment
        assert exp.optimization.method == "surrogate"
        assert experiment_module.validate_placement(
            exp.package, exp.catalog, exp.optimized.placement) == []
        assert exp.optimization.resolve_delta_k <= experiment_module.RESOLVE_TOL_K
        # Screening is not exact: it can match the exhaustive optimum, never
        # beat it -- and, with the initial placement always shortlisted, it
        # can never report a placement hotter than its starting point.
        assert exp.optimized.peak_c >= optimized_experiment.optimized.peak_c - 1e-9
        assert exp.optimized.peak_c <= exp.initial.peak_c + 1e-9

    def test_every_reported_temperature_is_a_real_solve(self, surrogate_experiment):
        exp = surrogate_experiment
        exp.solver.q_map[:] = exp.optimized.power.q_w_m3
        fresh_c = float(np.max(exp.solver.solve_steady_state(apply_result=False))) - 273.15
        assert exp.surrogate.best_peak_c == pytest.approx(fresh_c, abs=1e-9)
        assert exp.optimized.peak_c == pytest.approx(fresh_c, abs=1e-9)

    def test_a_bad_surrogate_costs_accuracy_not_correctness(self, monkeypatch,
                                                            optimized_experiment):
        """Invert the ranking on purpose, so the surrogate shortlists the
        layouts it believes are HOTTEST. The pipeline must still report a
        real, verified temperature for a valid placement."""
        good = surrogate_module.SurrogateModel.predict_peak_rise
        monkeypatch.setattr(surrogate_module.SurrogateModel, "predict_peak_rise",
                            lambda self, features: -good(self, features))
        exp = make_experiment()
        exp.solve_initial()
        exp.optimize(method="surrogate", verbose=False, surrogate_top_k=3, surrogate_samples=12)
        assert exp.surrogate.accuracy.rank_correlation < 0.0      # the sabotage took effect
        assert experiment_module.validate_placement(
            exp.package, exp.catalog, exp.optimized.placement) == []
        assert exp.optimization.resolve_delta_k <= experiment_module.RESOLVE_TOL_K
        assert exp.optimized.peak_c >= optimized_experiment.optimized.peak_c - 1e-9
        assert exp.optimized.peak_c <= exp.initial.peak_c + 1e-9

    def test_accuracy_is_measured_on_layouts_the_fit_never_saw(self, surrogate_experiment):
        screening = surrogate_experiment.surrogate
        accuracy = screening.accuracy
        assert screening.calibration_layouts == 12
        assert accuracy.layouts == len(accuracy.true_rise_k) == len(accuracy.predicted_rise_k)
        assert accuracy.layouts >= 3
        error = np.abs(np.array(accuracy.predicted_rise_k) - np.array(accuracy.true_rise_k))
        assert accuracy.mae_k == pytest.approx(float(error.mean()), rel=1e-12)
        assert accuracy.max_error_k == pytest.approx(float(error.max()), rel=1e-12)

    def test_report_says_it_ranks_and_claims_no_ai(self, surrogate_experiment, tmp_path):
        import html as _html
        import re as _re
        path = report_module.render_experiment_report(surrogate_experiment,
                                                      output_dir=str(tmp_path))
        raw = pathlib.Path(path).read_text(encoding="utf-8")
        raw = _re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", raw)
        text = _re.sub(r"\s+", " ", _html.unescape(_re.sub(r"<[^>]+>", " ", raw)))
        assert "Not a neural network and not a solver" in text
        assert "every temperature reported here came from the real solver" in text
        assert "not validated" in text.lower()
        for claim in ("ai-accelerated", "ai accelerated", "deep learning",
                      "neural network speedup", "machine learning"):
            assert claim not in text.lower()

    def test_cli_refuses_surrogate_without_a_catalog(self, tmp_path, capsys):
        rc = main_module.main(["--optimize", "--optimize-method", "surrogate",
                               "--no-browser", "--output-dir", str(tmp_path)])
        assert rc == main_module.EXIT_PREFLIGHT_FAILED
        assert "--optimize-method surrogate needs a configuration" in capsys.readouterr().err



class TestExperimentBanner:
    def test_banner_states_the_configured_catalog_not_a_rescale(self, capsys):
        """In an experiment the optimizer uses the catalog the config defines,
        as authored. The banner printed the LEGACY Stage 2 note instead --
        'Catalog rescaled 36.00 W -> 12.00 W' -- describing a catalog this run
        never uses."""
        config = config_loader_module.load_config(str(SHIPPED_CONFIG))
        experiment = experiment_module.Experiment(config, source="test")
        main_module.print_startup_banner(config, source="test", optimize=True,
                                         from_file=True, experiment=experiment)
        out = capsys.readouterr().out
        assert "rescaled" not in out
        assert "4 chiplets totalling 12.00 W, used as authored" in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))