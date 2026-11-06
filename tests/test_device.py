"""
tests/test_device.py
ThermalForge :: the device layer, and the loop it closes with the solver.

Two things are pinned here, and they fail differently.

The device model is held to arithmetic and to physics: the square law exactly
as written at the reference temperature, continuity where level 1 is usually
discontinuous, and the zero-temperature-coefficient crossing, which is not a
line of code anywhere -- it falls out of a falling threshold competing with a
falling mobility, and if either dependence were dropped or mis-signed the
crossing would vanish.

The loop is held to one rule, the same rule the AI layer is held to: the
acceleration proposes, the solver decides. Acceleration may change how many
passes a fixed point costs; it may not change which fixed point. That is
tested directly, by finding the same point with acceleration off.
"""

from __future__ import annotations

import math
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import studio.engine as E
import src.electrothermal as ET
from src.device import (KELVIN, MAX_JUNCTION_C, DeviceError, LinearSwitch,
                        Mosfet)


def nmos(**kw):
    base = dict(vth0_v=0.4, kp_a_v2=200e-6, w_over_l=1e5, lambda_1_v=0.0)
    base.update(kw)
    return Mosfet(**base)


@pytest.fixture(scope="module")
def cache():
    return E.ModelCache(capacity=2)


def one_part(power_w=4.0, size=10.0, h=2500.0):
    pkg = {"size_mm": size, "h_w_m2k": h, "ambient_c": 25.0, "pitch_mm": 2.0}
    chips = [{"id": "p0", "name": "P0", "w_mm": 3.0, "h_mm": 3.0,
              "power_w": power_w, "x_mm": 0.0, "y_mm": 0.0},
             {"id": "p1", "name": "P1", "w_mm": 3.0, "h_mm": 3.0,
              "power_w": 2.0, "x_mm": 6.0, "y_mm": 6.0}]
    spec = E.PackageSpec.parse(pkg)
    return spec, E.parse_chiplets(chips, spec)


# ---------------------------------------------------------------------------
# The square law, as written
# ---------------------------------------------------------------------------

class TestSquareLaw:

    def test_saturation_matches_the_closed_form_at_the_reference(self):
        m = nmos()
        vgs, vds = 1.0, 3.0
        vov = vgs - m.vth0_v
        expected = 0.5 * (m.kp_a_v2 * m.w_over_l) * vov * vov
        assert m.id_a(vgs, vds, m.t_ref_c) == pytest.approx(expected, rel=1e-12)

    def test_triode_matches_the_closed_form_at_the_reference(self):
        m = nmos()
        vgs, vds = 1.0, 0.2
        vov = vgs - m.vth0_v
        expected = (m.kp_a_v2 * m.w_over_l) * (vov * vds - 0.5 * vds * vds)
        assert m.id_a(vgs, vds, m.t_ref_c) == pytest.approx(expected, rel=1e-12)

    def test_it_is_continuous_across_the_region_boundary(self):
        # Level 1 with channel-length modulation applied in saturation only has
        # a step here. This model applies it in both regions precisely so the
        # current a design is handed does not jump as a bias crosses over.
        for lam in (0.0, 0.02, 0.2):
            m = nmos(lambda_1_v=lam)
            vgs = 1.0
            vov = vgs - m.vth_v(25.0)
            lo = m.id_a(vgs, vov * (1 - 1e-9), 25.0)
            hi = m.id_a(vgs, vov * (1 + 1e-9), 25.0)
            assert lo == pytest.approx(hi, rel=1e-9, abs=1e-12)

    def test_cutoff_is_exactly_zero_and_says_so(self):
        m = nmos()
        assert m.id_a(0.3, 1.0, 25.0) == 0.0
        assert m.power_w(0.3, 1.0, 25.0) == 0.0
        assert m.rds_on_ohm(0.3, 1.0, 25.0) == math.inf
        assert m.saturates_at_v(0.3, 25.0) < 0.0

    def test_power_is_vds_times_id_and_nothing_else(self):
        m = nmos(lambda_1_v=0.05)
        for vgs, vds, t in ((1.0, 0.3, 25.0), (1.2, 2.0, 95.0), (0.9, 0.5, 125.0)):
            assert m.power_w(vgs, vds, t) == pytest.approx(
                vds * m.id_a(vgs, vds, t), rel=1e-14)

    def test_operating_point_resistance_tends_to_the_small_signal_slope(self):
        # As Vds -> 0 the triode law linearises to Id = beta * Vov * Vds, so
        # Vds / Id -> 1 / (beta Vov). Checked at 1 mV.
        m = nmos()
        vgs, t = 1.0, 25.0
        vov = vgs - m.vth_v(t)
        expected = 1.0 / (m.beta_a_v2(t) * vov)
        assert m.rds_on_ohm(vgs, 1e-3, t) == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# The two temperature dependences, and the crossing between them
# ---------------------------------------------------------------------------

class TestTemperature:

    def test_threshold_falls_at_the_stated_slope(self):
        m = nmos(dvth_dt_v_k=-2.0e-3)
        assert m.vth_v(25.0) == pytest.approx(0.4)
        assert m.vth_v(125.0) == pytest.approx(0.4 - 0.2)
        assert m.vth_v(125.0) < m.vth_v(25.0)

    def test_mobility_falls_as_the_stated_power_of_absolute_temperature(self):
        m = nmos(mu_exponent=1.5)
        ratio = (125.0 + KELVIN) / (25.0 + KELVIN)
        assert m.beta_a_v2(125.0) == pytest.approx(
            m.beta_a_v2(25.0) * ratio ** -1.5, rel=1e-12)
        assert m.beta_a_v2(125.0) < m.beta_a_v2(25.0)

    def test_near_threshold_the_current_rises_with_temperature(self):
        # The threshold term wins: this is the bias that runs away.
        m = nmos()
        assert m.id_a(0.6, 3.0, 125.0) > m.id_a(0.6, 3.0, 25.0)

    def test_well_above_threshold_the_current_falls_with_temperature(self):
        # The mobility term wins: this bias is self-limiting.
        m = nmos()
        assert m.id_a(2.0, 3.0, 125.0) < m.id_a(2.0, 3.0, 25.0)

    def test_there_is_a_zero_temperature_coefficient_bias_between_them(self):
        # Nothing in the model is told to produce this. It exists because the
        # two dependences have opposite sign, which is how silicon behaves.
        m = nmos()

        def drift(vgs):
            return m.id_a(vgs, 3.0, 125.0) - m.id_a(vgs, 3.0, 25.0)

        lo, hi = 0.6, 2.0
        assert drift(lo) > 0.0 and drift(hi) < 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if drift(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        ztc = 0.5 * (lo + hi)
        assert 0.6 < ztc < 2.0
        assert drift(ztc) == pytest.approx(0.0, abs=1e-6)

    def test_it_refuses_a_temperature_it_does_not_describe(self):
        m = nmos()
        with pytest.raises(DeviceError, match="extrapolating"):
            m.id_a(1.0, 1.0, MAX_JUNCTION_C + 1.0)
        with pytest.raises(DeviceError, match="absolute zero"):
            m.id_a(1.0, 1.0, -300.0)


# ---------------------------------------------------------------------------
# Parameters it will not accept
# ---------------------------------------------------------------------------

class TestRefusals:

    @pytest.mark.parametrize("kw", [
        {"vth0_v": 0.0}, {"vth0_v": -0.4}, {"kp_a_v2": 0.0},
        {"w_over_l": -1.0}, {"lambda_1_v": -0.1}, {"mu_exponent": -1.0},
        {"dvth_dt_v_k": float("nan")},
    ])
    def test_a_parameter_that_is_not_a_device_is_refused(self, kw):
        with pytest.raises(DeviceError):
            nmos(**kw)

    def test_a_negative_bias_is_refused_rather_than_silently_signed(self):
        m = nmos()
        with pytest.raises(DeviceError, match="vds_v"):
            m.id_a(1.0, -1.0, 25.0)
        with pytest.raises(DeviceError, match="vgs_v"):
            m.id_a(-1.0, 1.0, 25.0)


# ---------------------------------------------------------------------------
# The simplified feedback path
# ---------------------------------------------------------------------------

class TestLinearSwitch:

    def test_resistance_is_linear_in_temperature(self):
        s = LinearSwitch(rds0_ohm=0.05, alpha_per_k=5e-3)
        assert s.rds_ohm(25.0) == pytest.approx(0.05)
        assert s.rds_ohm(125.0) == pytest.approx(0.05 * 1.5)

    def test_power_is_i_squared_r(self):
        s = LinearSwitch(rds0_ohm=0.05, alpha_per_k=5e-3)
        assert s.power_w(10.0, 125.0) == pytest.approx(100.0 * s.rds_ohm(125.0))
        assert s.vds_v(10.0, 125.0) == pytest.approx(10.0 * s.rds_ohm(125.0))

    def test_it_refuses_the_temperature_where_the_linear_law_goes_negative(self):
        # A linear drift extrapolated far enough predicts a negative
        # resistance. That is the model leaving its range, not a device.
        s = LinearSwitch(rds0_ohm=1.0, alpha_per_k=-0.02)
        assert s.rds_ohm(50.0) > 0.0
        with pytest.raises(DeviceError, match="not physical"):
            s.rds_ohm(100.0)

    def test_bad_parameters_are_refused(self):
        with pytest.raises(DeviceError):
            LinearSwitch(rds0_ohm=0.0)
        with pytest.raises(DeviceError):
            LinearSwitch(rds0_ohm=1.0, alpha_per_k=float("inf"))


# ---------------------------------------------------------------------------
# The loop: power at an assumed junction -> solve -> junction
# ---------------------------------------------------------------------------

class TestCoupledLoop:

    def test_it_settles_and_the_answer_is_self_consistent(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        assert out["converged"] is True
        assert out["pass_count"] <= ET.DEFAULT_MAX_PASSES
        # The fixed point: the power reported is what the device draws at the
        # junction reported, and that junction came out of a solve.
        tj = out["junction_c"]["p0"]
        assert out["power_w"]["p0"] == pytest.approx(src.power_w(tj), rel=1e-12)
        last = out["passes"][-1]
        assert last["delta_k"] <= out["tol_k"]
        assert out["solve"]["ok"] is True

    def test_a_part_with_no_source_keeps_the_power_it_was_given(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        rows = {r["id"]: r for r in out["solve"]["chiplets"]}
        assert rows["p1"]["power_w"] == pytest.approx(2.0)
        assert "p1" not in out["power_w"]

    def test_geometry_is_never_touched_by_the_power_layer(self):
        spec, parts = one_part()
        swapped = ET._with_powers(parts, {"p0": 99.0})
        for before, after in zip(parts, swapped):
            assert (after.w_mm, after.h_mm, after.x_mm, after.y_mm, after.id) == \
                   (before.w_mm, before.h_mm, before.x_mm, before.y_mm, before.id)
            assert after.power_pattern == before.power_pattern
        assert swapped[0].power_w == 99.0

    def test_positive_feedback_raises_the_power_above_its_cold_value(self, cache):
        spec, parts = one_part()
        sw = LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3)
        src = ET.SwitchSource("p0", sw, current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        cold = src.power_w(spec.ambient_c)
        assert out["power_w"]["p0"] > cold
        assert out["loop_gain"]["p0"]["dp_dt_w_k"] > 0.0
        assert out["loop_gain"]["p0"]["stable"] is True

    def test_negative_feedback_lowers_it(self, cache):
        spec, parts = one_part()
        # Biased above the ZTC point, so hotter draws less.
        m = nmos(w_over_l=4000)
        src = ET.MosfetSource("p0", m, vgs_v=2.0, vds_v=3.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        assert out["converged"] is True
        assert out["power_w"]["p0"] < src.power_w(spec.ambient_c)
        assert out["loop_gain"]["p0"]["dp_dt_w_k"] < 0.0

    def test_acceleration_changes_the_cost_and_not_the_fixed_point(self, cache):
        # The one rule: extrapolation proposes where to evaluate, the solver
        # decides what the temperature is. Same point, found two ways.
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        fast = ET.solve_coupled(cache, spec, parts, [src], accelerate=True)
        slow = ET.solve_coupled(cache, spec, parts, [src], accelerate=False,
                                max_passes=40, tol_k=1e-4)
        assert fast["converged"] and slow["converged"]
        assert fast["pass_count"] < slow["pass_count"]
        assert fast["junction_c"]["p0"] == pytest.approx(
            slow["junction_c"]["p0"], abs=2 * fast["tol_k"])
        assert fast["power_w"]["p0"] == pytest.approx(
            slow["power_w"]["p0"], rel=1e-3)

    def test_a_runaway_is_refused_rather_than_reported(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=8e-3),
                              current_a=30.0)
        with pytest.raises(ET.ThermalRunaway):
            ET.solve_coupled(cache, spec, parts, [src])

    def test_every_reported_temperature_came_out_of_a_solve(self, cache):
        # No pass may report a junction that the solver did not produce.
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        for p in out["passes"]:
            assert set(p["junction_c"]) == {"p0"}
            assert p["peak_c"] >= p["junction_c"]["p0"] - 1e-9
        assert out["junction_c"] == out["passes"][-1]["junction_c"]

    def test_the_loop_gain_is_reported_with_its_two_factors(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        g = out["loop_gain"]["p0"]
        assert g["loop_gain"] == pytest.approx(g["dp_dt_w_k"] * g["r_th_k_w"],
                                               rel=1e-12)
        # dP/dT for I^2 R(T) is I^2 R0 alpha, exactly and independent of T.
        assert g["dp_dt_w_k"] == pytest.approx(81.0 * 0.05 * 4e-3, rel=1e-6)

    @pytest.mark.parametrize("bad,match", [
        ("unknown", "not in this placement"),
        ("duplicate", "same chiplet"),
        ("none", "no device sources"),
    ])
    def test_inputs_that_do_not_describe_a_loop_are_refused(self, cache, bad, match):
        spec, parts = one_part()
        sw = LinearSwitch(rds0_ohm=0.05)
        if bad == "unknown":
            sources = [ET.SwitchSource("nope", sw, current_a=5.0)]
        elif bad == "duplicate":
            sources = [ET.SwitchSource("p0", sw, current_a=5.0),
                       ET.SwitchSource("p0", sw, current_a=6.0)]
        else:
            sources = []
        with pytest.raises(ET.CouplingError, match=match):
            ET.solve_coupled(cache, spec, parts, sources)

    def test_a_source_needs_at_least_one_device(self):
        with pytest.raises(DeviceError, match="count"):
            ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05), current_a=1.0, count=0)
        with pytest.raises(DeviceError, match="count"):
            ET.MosfetSource("p0", nmos(), vgs_v=1.0, vds_v=1.0, count=0)

    def test_device_count_scales_the_power_linearly(self):
        sw = LinearSwitch(rds0_ohm=0.05)
        one = ET.SwitchSource("p0", sw, current_a=5.0, count=1)
        ten = ET.SwitchSource("p0", sw, current_a=5.0, count=10)
        assert ten.power_w(80.0) == pytest.approx(10.0 * one.power_w(80.0))

    def test_summary_lines_state_whether_it_settled(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        text = "\n".join(ET.summary_lines(out))
        assert "settled" in text
        assert "loop gain" in text
        assert "modelled rather than measured" in text


# ---------------------------------------------------------------------------
# Running out of passes is not the same failure as running away
# ---------------------------------------------------------------------------

class TestBudgetVersusRunaway:
    """A loop whose gain is below 1 has a steady state whether or not five
    passes were enough to reach it. Reporting the two as one thing would send
    a design back for cooling it does not need, so they are kept apart and the
    cheaper one is quantified: how many passes it would actually take.
    """

    @pytest.mark.parametrize("current,expected_passes", [(8.5, 7), (9.0, 9)])
    def test_it_predicts_the_passes_it_would_need(self, cache, current,
                                                  expected_passes):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=8e-3),
                              current_a=current)
        short = ET.solve_coupled(cache, spec, parts, [src], max_passes=5)
        assert short["converged"] is False
        assert short["loop_gain"]["p0"]["stable"] is True   # not a runaway
        # The prediction comes from the two moves it made, and it is checked
        # against what the loop actually costs when the budget is lifted.
        assert short["passes_needed_estimate"] == expected_passes
        full = ET.solve_coupled(cache, spec, parts, [src], max_passes=30)
        assert full["converged"] is True
        assert full["pass_count"] == expected_passes

    def test_running_out_of_passes_does_not_raise(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=8e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src], max_passes=3)
        assert out["ok"] is True and out["converged"] is False

    def test_a_settled_loop_predicts_nothing_because_it_needs_nothing(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=4e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src])
        assert out["converged"] is True
        assert out["passes_needed_estimate"] is None

    def test_the_summary_distinguishes_the_two(self, cache):
        spec, parts = one_part()
        src = ET.SwitchSource("p0", LinearSwitch(rds0_ohm=0.05, alpha_per_k=8e-3),
                              current_a=9.0)
        out = ET.solve_coupled(cache, spec, parts, [src], max_passes=5)
        text = "\n".join(ET.summary_lines(out))
        assert "DID NOT SETTLE" in text
        assert "rather than a runaway" in text
        assert "about 9 passes" in text
