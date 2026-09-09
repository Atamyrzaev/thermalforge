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

from typing import Optional

import numpy as np
import pytest

from src.config_loader import (
    PipelineConfig,
    ConfigError,
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

_EPS = 1e-30


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def make_pipeline_config(
    nx: int = 10, ny: int = 10, nz: int = 12,
    dx: float = 25e-6, dy: float = 25e-6, dz: float = 10e-6,
    t_ambient_c: float = 25.0, h_convective: float = 2500.0, bottom_bc: str = "robin",
    interposer_um: Optional[float] = None, bump_um: Optional[float] = None,
    die_um: Optional[float] = None,
    die_power_w: float = 0.05, die_footprint_frac: float = 0.5,
) -> PipelineConfig:
    """Builds a small, fast-to-solve PipelineConfig for unit testing.

    Stack layers are authored in MICROMETERS (see StackSection). When not
    given explicitly they are derived from the grid's own z-extent in the
    40%/20%/40% proportion the old fraction-based defaults produced, so the
    authored stack always matches nz*dz exactly and these unit tests never
    trip the grid/stack mismatch warning."""
    total_um = nz * dz * 1e6
    if interposer_um is None:
        interposer_um = 0.4 * total_um
    if bump_um is None:
        bump_um = 0.2 * total_um
    if die_um is None:
        die_um = total_um - interposer_um - bump_um
    return PipelineConfig(
        grid=GridSection(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz),
        boundary_conditions=BoundaryConditionsSection(
            t_ambient_c=t_ambient_c, h_convective=h_convective, bottom_bc=bottom_bc,
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

    if gb.config.bottom_bc == "dirichlet":
        coeff_b = 2.0 * kz[:, :, 0] / dz ** 2
    else:
        coeff_b = np.full((nx, ny), gb.config.h_convective / dz)
    total[:, :, 0] += coeff_b

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
                "bottom_bc": "robin", "top_bc": "neumann", "side_bc": "neumann",
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
        config = make_pipeline_config(bottom_bc=bottom_bc, die_power_w=0.0)
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
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin")
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
        config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0, h_convective=1800.0)
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
        config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0)
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
            config = make_pipeline_config(bottom_bc="robin", die_power_w=0.0, h_convective=h_value)
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
        config = make_pipeline_config(bottom_bc="dirichlet", die_power_w=0.0)
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
        config = make_pipeline_config(die_power_w=0.08, bottom_bc="robin")
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
        boundary_outflow = float(np.sum(coeff_b * (T_steady[:, :, 0] - solver.bc.t_ambient_k)))
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

        # A uniform bottom-layer offset breaks the energy balance.
        corrupted2 = T.copy()
        corrupted2[:, :, 0] += 5.0
        with pytest.raises(ThermalSolverError):
            solver._verify_steady_state(corrupted2.reshape(N), forcing)

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
    whole die and leaves no placeable positions."""
    return ChipletPlacementOptimizer(
        nx=20, ny=20, nz=12, dx=25e-6, dy=25e-6, dz=10e-6,
        catalog=SMALL_CATALOG, t_ambient_c=25.0, h_convective=2500.0,
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

        # 40x40 at 500 um = 20 x 20 mm = 400 mm^2 die -> accepted. nz=12 here
        # (19,200 unknowns, direct sparse path) rather than 24: at 38,400 this
        # mesh falls to the iterative backend, which does not converge on a
        # grid this anisotropic. Only construction and geometry matter to this
        # test, but there is no reason for it to stand on a configuration the
        # module documents as unusable.
        opt = ChipletPlacementOptimizer(nx=40, ny=40, nz=12,
                                        dx=500e-6, dy=500e-6, dz=40e-6)
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
        rise = float(np.mean(T[:, :, 0])) - solver.bc.t_ambient_k
        expected = self._analytic_mean_bottom_rise(config, config.power.die_power_w)

        assert rise == pytest.approx(expected, rel=1e-6)

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
        rise = float(np.mean(T_iter[:, :, 0])) - solver_iter.bc.t_ambient_k
        assert rise == pytest.approx(
            self._analytic_mean_bottom_rise(config, config.power.die_power_w), rel=1e-6
        )

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


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))