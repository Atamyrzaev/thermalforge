"""
tests/test_studio.py
RISC3D Studio: the engine, its exact optimizer, and the HTTP API.

The central claim of the product is that the optimizer returns the TRUE
coolest layout on the placement grid. These tests check that claim against
an independent brute force: every layout enumerated, each scored from
single-chiplet float64 fields from the verified solver, over ALL cells --
sharing no code with the branch-and-bound search.
"""

from __future__ import annotations

import base64
import itertools
import json
import math
import pathlib
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from studio import engine as E
from studio.scenarios import scenarios
from studio.server import serve


def spec(**overrides) -> E.PackageSpec:
    raw = {"size_mm": 10.0, "h_w_m2k": 2000.0, "ambient_c": 25.0, "pitch_mm": 2.0}
    raw.update(overrides)
    return E.PackageSpec.parse(raw)


def chip(cid, w, h, power, x, y):
    return {"id": cid, "name": cid.upper(), "w_mm": w, "h_mm": h, "power_w": power, "x_mm": x, "y_mm": y}


def field_c(payload) -> np.ndarray:
    g = payload["grid"]
    data = np.frombuffer(base64.b64decode(payload["field_b64"]), dtype="<f4")
    return data.reshape(g["nx"], g["ny"], g["nz"]).astype(np.float64)


@pytest.fixture(scope="module")
def cache():
    return E.ModelCache()


def brute_force_optimum(cache, pkg, chiplets):
    """Every non-overlapping grid layout, scored over every cell from
    single-chiplet solves: full float64 fields straight from the verified
    solver (src.experiment.solve_placement), sharing no code with the
    branch-and-bound search or its unit fields. Returns (peak degC, xy)."""
    from src.experiment import solve_placement
    model, _ = cache.get(pkg)
    pitch, size = pkg.pitch_mm, pkg.size_mm
    options = []
    for c in chiplets:
        spots = []
        for x in np.arange(0.0, size - c["w_mm"] + 1e-9, pitch):
            for y in np.arange(0.0, size - c["h_mm"] + 1e-9, pitch):
                single = E.parse_chiplets([{**c, "x_mm": float(x), "y_mm": float(y)}], pkg)
                with model.lock:
                    result = solve_placement(model.solver, model.package, E._catalog(single),
                                             E._placement(single, pkg.cell_mm), "brute force")
                rise = np.asarray(result.field_k, dtype=np.float64) - model.package.t_ambient_k
                spots.append(((float(x), float(y)), rise))
        options.append(spots)
    # The same assembly rule the engine applies, written independently here:
    # a pair is legal when it clears the keep-out in x or in y.
    keepout = pkg.keepout_enforced_um * 1e-3
    best, best_xy = np.inf, None
    for combo in itertools.product(*options):
        rects = [(xy[0], xy[1], c["w_mm"], c["h_mm"]) for (xy, _), c in zip(combo, chiplets)]
        too_close = any(min(a[0] + a[2], b[0] + b[2]) + keepout - max(a[0], b[0]) > 1e-9
                        and min(a[1] + a[3], b[1] + b[3]) + keepout - max(a[1], b[1]) > 1e-9
                        for a, b in itertools.combinations(rects, 2))
        if too_close:
            continue
        peak = float(sum(rise for _, rise in combo).max()) + pkg.ambient_c
        if peak < best:
            best, best_xy = peak, [xy for xy, _ in combo]
    return best, best_xy


class TestEngineSolve:
    def test_energy_is_conserved_and_the_floor_holds(self, cache):
        pkg = spec()
        chiplets = E.parse_chiplets([chip("a", 3, 3, 5.0, 0, 0), chip("b", 2, 4, 3.0, 6, 5)], pkg)
        out = E.solve(cache, pkg, chiplets)
        assert out["ok"]
        v, m = out["verification"], out["metrics"]
        assert v["deposited_w"] == pytest.approx(8.0, rel=1e-12)
        assert v["sink_heat_w"] == pytest.approx(8.0, rel=1e-9)
        assert v["residual"] < 1e-8
        assert m["rise_k"] > m["floor_rise_k"]
        assert out["peak"]["t_c"] == pytest.approx(field_c(out).max(), abs=1e-4)

    def test_symmetric_factor_matches_the_solvers_own_system(self, cache):
        model, _ = cache.get(spec())
        solver = model.solver
        n = solver.nx * solver.ny * solver.nz
        rhs = np.random.default_rng(1).random(n)
        reference = spla.splu(solver._conduction_operator.tocsc()).solve(rhs)
        assert np.max(np.abs(model.factor.solve(rhs) - reference)) <= 1e-9 * np.max(np.abs(reference))

    def test_overlapping_placement_is_reported_not_solved(self, cache):
        pkg = spec()
        chiplets = E.parse_chiplets([chip("a", 4, 4, 5.0, 0, 0), chip("b", 4, 4, 5.0, 2, 2)], pkg)
        out = E.solve(cache, pkg, chiplets)
        assert out["ok"] is False
        assert any("overlap" in v for v in out["violations"])

    @pytest.mark.parametrize("raw, message", [
        ({"size_mm": 10, "h_w_m2k": 2000, "ambient_c": 25, "pitch_mm": 0.7}, "whole number"),
        ({"size_mm": 10, "h_w_m2k": 0, "ambient_c": 25, "pitch_mm": 2}, "h_w_m2k"),
        ({"size_mm": "ten", "h_w_m2k": 2000, "ambient_c": 25, "pitch_mm": 2}, "size_mm"),
    ])
    def test_bad_package_input_is_refused(self, raw, message):
        with pytest.raises(E.StudioError, match=message):
            E.PackageSpec.parse(raw)

    def test_bad_chiplet_input_is_refused(self):
        pkg = spec()
        with pytest.raises(E.StudioError, match="duplicate"):
            E.parse_chiplets([chip("a", 2, 2, 1, 0, 0), chip("a", 2, 2, 1, 4, 4)], pkg)
        with pytest.raises(E.StudioError, match="grid"):
            E.parse_chiplets([chip("a", 2.3, 2, 1, 0, 0)], pkg)
        with pytest.raises(E.StudioError, match="at least one"):
            E.parse_chiplets([], pkg)


class TestLauncher:
    """The entry point every packaged build runs. Nothing else imports it, so
    without this a typo in it survives a green test suite and only shows up
    when the exe is double-clicked."""

    def test_the_launcher_parses_and_exposes_main(self):
        import ast
        import inspect
        import run_studio
        source = pathlib.Path(run_studio.__file__).read_text(encoding="utf-8")
        ast.parse(source)                      # a syntax error here breaks the build only
        assert callable(run_studio.main)
        assert "argv" in inspect.signature(run_studio.main).parameters

    def test_it_refuses_a_port_it_cannot_have_instead_of_crashing(self, base_url):
        """The port in use is the one the fixture's server is already on."""
        import run_studio
        port = int(base_url.rsplit(":", 1)[1])
        assert run_studio.main(["--port", str(port), "--no-browser"]) == 1


class TestLauncher:
    """The entry point every packaged build runs. Nothing else imports it, so
    without this a typo in it survives a green test suite and shows up only
    when someone double-clicks the app."""

    def test_the_launcher_parses_and_exposes_main(self):
        import ast
        import inspect
        import pathlib
        import run_studio
        ast.parse(pathlib.Path(run_studio.__file__).read_text(encoding="utf-8"))
        assert callable(run_studio.main)
        assert "argv" in inspect.signature(run_studio.main).parameters

    def test_a_second_copy_joins_the_first_instead_of_fighting_for_the_port(self, base_url):
        """Windows lets a second process bind a port the first one holds, and
        requests then land on whichever socket wins. Double-clicking twice at
        a trade show must open the running app, not a confused second one."""
        import run_studio
        host, port = base_url.rsplit(":", 1)
        assert run_studio._is_in_use("127.0.0.1", int(port)) is True
        assert run_studio.main(["--port", port, "--no-browser"]) == 0

    def test_a_free_port_is_not_mistaken_for_a_running_app(self):
        import socket
        import run_studio
        with socket.socket() as probe:          # bind, read the port, release it
            probe.bind(("127.0.0.1", 0))
            free = probe.getsockname()[1]
        assert run_studio._is_in_use("127.0.0.1", free) is False


class TestCoolingPath:
    """Heat does not leave a real package through a film coefficient on bare
    silicon: it crosses a TIM into a lid or a cold plate, SPREADS sideways in
    that metal, and only then meets the coolant. The spreading is why the
    layers are meshed instead of folded into an effective h."""

    CHIPLET = [{"id": "npu", "name": "NPU", "w_mm": 5, "h_mm": 5, "power_w": 8,
                "x_mm": 7, "y_mm": 7, "power_pattern": "mac_array"}]

    def peak(self, cache, cooling, **overrides):
        pkg = spec(size_mm=20.0, h_w_m2k=1000.0, cooling=cooling, **overrides)
        out = E.solve(cache, pkg, E.parse_chiplets(self.CHIPLET, pkg))
        assert out["ok"], out.get("violations")
        return out

    def test_a_lid_spreads_heat_and_the_floor_does_not_move(self, cache):
        """The cooling floor is P/(hA): same power, same outer area, same
        floor. A lid cannot beat it -- what a lid removes is the SPREADING
        penalty on top of it, which is most of the hotspot."""
        bare = self.peak(cache, "direct")
        lidded = self.peak(cache, "lid")
        assert lidded["metrics"]["floor_rise_k"] == pytest.approx(bare["metrics"]["floor_rise_k"])
        assert lidded["peak"]["t_c"] < bare["peak"]["t_c"] - 20.0
        assert lidded["peak"]["t_c"] - lidded["ambient_c"] > lidded["metrics"]["floor_rise_k"]

    def test_the_path_is_meshed_and_the_die_is_still_the_die(self, cache):
        """The extra cells are cooling, not silicon: power still goes into
        the die's active plane and the peak is still reported in the die."""
        out = self.peak(cache, "lid")
        g = out["grid"]
        assert g["nz"] == 15                       # 12 stack + 1 TIM + 2 lid
        assert [layer["name"] for layer in g["layers"]][-2:] == ["TIM", "Copper lid"]
        assert (g["die_k0"], g["die_k1"]) == (7, 12)
        assert out["peak"]["layer"] == "Die (Si)"
        model, _ = cache.get(spec(size_mm=20.0, h_w_m2k=1000.0, cooling="lid"))
        assert model.power_z == (7, 8)             # the active plane, unmoved

    def test_a_thicker_lid_spreads_more(self, cache):
        thin = spec(size_mm=20.0, h_w_m2k=1000.0,
                    cooling_layers=[["TIM", 50.0, 5.0, 1], ["Lid", 500.0, 400.0, 2]])
        thick = spec(size_mm=20.0, h_w_m2k=1000.0,
                     cooling_layers=[["TIM", 50.0, 5.0, 1], ["Lid", 2000.0, 400.0, 2]])
        hot = E.solve(cache, thin, E.parse_chiplets(self.CHIPLET, thin))
        cool = E.solve(cache, thick, E.parse_chiplets(self.CHIPLET, thick))
        assert cool["peak"]["t_c"] < hot["peak"]["t_c"]

    def test_a_worse_tim_costs_temperature(self, cache):
        """The TIM is in series with everything: its resistance lands on
        every watt, so a worse one can only make the die hotter."""
        good = spec(size_mm=20.0, h_w_m2k=1000.0,
                    cooling_layers=[["TIM", 50.0, 8.0, 1], ["Lid", 1000.0, 400.0, 2]])
        poor = spec(size_mm=20.0, h_w_m2k=1000.0,
                    cooling_layers=[["TIM", 50.0, 1.0, 1], ["Lid", 1000.0, 400.0, 2]])
        assert (E.solve(cache, poor, E.parse_chiplets(self.CHIPLET, poor))["peak"]["t_c"]
                > E.solve(cache, good, E.parse_chiplets(self.CHIPLET, good))["peak"]["t_c"])

    def test_the_compact_layer_keeps_the_through_plane_resistance(self, cache):
        """A 1 mm lid modelled as 2 cells must still impose t/k of the real
        lid. Checked against the closed form: with the same spreading in both
        cases (one uniform source over the whole package), the difference
        between a bare die and a lidded one is exactly the layers' 1-D
        resistance."""
        area_m2 = (20e-3) ** 2
        # One chiplet covering the whole package: no spreading anywhere, so
        # every remaining difference is 1-D resistance.
        flat = [{"id": "all", "name": "ALL", "w_mm": 20, "h_mm": 20, "power_w": 8,
                 "x_mm": 0, "y_mm": 0}]
        bare = spec(size_mm=20.0, h_w_m2k=1000.0, cooling="direct")
        lidded = spec(size_mm=20.0, h_w_m2k=1000.0,
                      cooling_layers=[["TIM", 50.0, 5.0, 1], ["Lid", 1000.0, 400.0, 2]])
        a = E.solve(cache, bare, E.parse_chiplets(flat, bare))["peak"]["t_c"]
        b = E.solve(cache, lidded, E.parse_chiplets(flat, lidded))["peak"]["t_c"]
        flux = 8.0 / area_m2
        expected = flux * (50e-6 / 5.0 + 1000e-6 / 400.0)      # t/k of TIM + lid
        assert b - a == pytest.approx(expected, rel=0.02)

    def test_an_impossible_cooling_path_is_refused(self):
        with pytest.raises(E.StudioError, match="cooling must be one of"):
            E.PackageSpec.parse({"size_mm": 10.0, "h_w_m2k": 2000.0, "ambient_c": 25.0,
                                 "pitch_mm": 2.0, "cooling": "magic"})
        with pytest.raises(E.StudioError, match="cells must be a whole number"):
            E.PackageSpec.parse({"size_mm": 10.0, "h_w_m2k": 2000.0, "ambient_c": 25.0,
                                 "pitch_mm": 2.0, "cooling_layers": [["Lid", 1000.0, 400.0, 0]]})
        with pytest.raises(E.StudioError, match="at most"):
            E.PackageSpec.parse({"size_mm": 10.0, "h_w_m2k": 2000.0, "ambient_c": 25.0,
                                 "pitch_mm": 2.0,
                                 "cooling_layers": [["Lid", 1000.0, 400.0, 6]] * 3})

    def test_the_optimizer_works_through_a_cooling_path(self, cache):
        pkg = spec(size_mm=20.0, h_w_m2k=1000.0, pitch_mm=4.0, cooling="lid")
        raw = [chip("a", 4, 4, 6.0, 0, 0), chip("b", 4, 4, 4.0, 12, 12)]
        out = E.optimize(cache, pkg, E.parse_chiplets(raw, pkg))
        assert out["certificate"]["resolve_delta_k"] <= E.RESOLVE_TOL_K
        assert out["optimized"]["peak"]["t_c"] <= out["initial"]["peak"]["t_c"] + 1e-9


class TestInternalPowerMap:
    """A chiplet is not a uniform heater. An NPU's MAC array, a GPU's engines
    and a CPU's core complex draw most of the current over part of the die,
    and the junction follows the hot region, not the average."""

    def test_the_same_watts_run_hotter_when_they_are_concentrated(self, cache):
        pkg = spec(size_mm=20.0, h_w_m2k=2500.0)
        base = {"id": "npu", "name": "NPU", "w_mm": 5, "h_mm": 5, "power_w": 8,
                "x_mm": 7, "y_mm": 7}
        peaks = {}
        for pattern in ("uniform", "quad_cores", "two_engines", "mac_array"):
            chiplets = E.parse_chiplets([dict(base, power_pattern=pattern)], pkg)
            out = E.solve(cache, pkg, chiplets)
            peaks[pattern] = out["peak"]["t_c"]
            # Conservation is not optional: the watts must still all arrive.
            v = out["verification"]
            assert v["deposited_w"] == pytest.approx(8.0, rel=1e-9)
            assert v["sink_heat_w"] == pytest.approx(8.0, rel=1e-9)
        assert peaks["uniform"] < peaks["quad_cores"] < peaks["two_engines"] < peaks["mac_array"]
        # And the effect is worth modelling: a MAC array is >10 K hotter than
        # the uniform assumption at identical power.
        assert peaks["mac_array"] - peaks["uniform"] > 10.0

    def test_the_hot_region_density_is_reported_not_just_the_average(self, cache):
        pkg = spec(size_mm=20.0)
        chiplets = E.parse_chiplets([{"id": "npu", "name": "NPU", "w_mm": 5, "h_mm": 5,
                                      "power_w": 8, "x_mm": 6, "y_mm": 6,
                                      "power_pattern": "mac_array"}], pkg)
        row = E.solve(cache, pkg, chiplets)["chiplets"][0]
        assert row["density_w_cm2"] == pytest.approx(32.0)          # 8 W over 25 mm^2
        assert row["hottest_density_w_cm2"] > 50.0                  # 70 % over 40 % of it
        assert row["power_pattern"] == "mac_array"

    def test_blocks_land_on_cells_and_never_round_away(self):
        """A block smaller than one cell still has to go somewhere: rounding
        it away would move its power back into the uniform part silently."""
        from src.experiment import PowerBlock, block_cells, uniform_share
        tiny = (PowerBlock(0.45, 0.45, 0.02, 0.02, 0.5),)
        assert block_cells(4, 4, tiny) == ((2, 2, 1, 1, 0.5),)
        assert uniform_share(tiny) == pytest.approx(0.5)
        # A block that would run past the edge is clipped to the footprint.
        wide = (PowerBlock(0.75, 0.0, 0.5, 1.0, 0.4),)
        i0, j0, w, h, share = block_cells(4, 4, wide)[0]
        assert i0 + w <= 4 and j0 + h <= 4 and share == 0.4

    def test_two_chiplets_of_one_size_but_different_maps_are_solved_apart(self, cache):
        """The optimizer shares single-chiplet solves between identical
        footprints. Identical must include what is inside them."""
        pkg = spec(size_mm=20.0)
        raw = [{"id": "a", "name": "A", "w_mm": 4, "h_mm": 4, "power_w": 6,
                "x_mm": 0, "y_mm": 0, "power_pattern": "mac_array"},
               {"id": "b", "name": "B", "w_mm": 4, "h_mm": 4, "power_w": 6,
                "x_mm": 10, "y_mm": 10, "power_pattern": "uniform"}]
        chiplets = E.parse_chiplets(raw, pkg)
        model, _ = cache.get(pkg)
        sizes = [(8, 8), (8, 8)]
        positions = [np.array([(0, 0), (20, 20)]), np.array([(0, 0), (20, 20)])]
        with model.lock:
            fields = E.unit_fields(model, chiplets, positions, sizes)
        assert not np.allclose(fields[0], fields[1]), "the two maps shared one solve"

    @pytest.mark.parametrize("blocks, message", [
        ([[0.1, 0.1, 0.5, 0.5, 0.7], [0.1, 0.1, 0.5, 0.5, 0.7]], "more than the chiplet has"),
        ([[0.8, 0.1, 0.5, 0.2, 0.3]], "past the edge"),
        ([[0.1, 0.1, 0.5, 0.5]], r"\[x, y, width, height, share\]"),
        ([[0.1, 0.1, 0.0, 0.5, 0.3]], "inside the footprint"),
    ])
    def test_an_impossible_internal_map_is_refused(self, blocks, message):
        pkg = spec()
        with pytest.raises(E.StudioError, match=message):
            E.parse_chiplets([{"id": "a", "name": "A", "w_mm": 4, "h_mm": 4, "power_w": 5,
                               "x_mm": 0, "y_mm": 0, "blocks": blocks}], pkg)

    def test_an_unknown_pattern_is_refused_with_the_list(self):
        pkg = spec()
        with pytest.raises(E.StudioError, match="power_pattern must be one of"):
            E.parse_chiplets([{"id": "a", "name": "A", "w_mm": 4, "h_mm": 4, "power_w": 5,
                               "x_mm": 0, "y_mm": 0, "power_pattern": "sprinkles"}], pkg)

    def test_the_optimizer_still_proves_its_answer_with_internal_maps(self, cache):
        """Superposition does not care where inside a chiplet the heat is, as
        long as the unit field is built the same way -- and the re-solve check
        is what proves it did."""
        pkg = spec(size_mm=20.0, pitch_mm=4.0)
        raw = [{"id": "npu", "name": "NPU", "w_mm": 4, "h_mm": 4, "power_w": 8,
                "x_mm": 0, "y_mm": 0, "power_pattern": "mac_array"},
               {"id": "cpu", "name": "CPU", "w_mm": 4, "h_mm": 4, "power_w": 5,
                "x_mm": 12, "y_mm": 12, "power_pattern": "quad_cores"}]
        out = E.optimize(cache, pkg, E.parse_chiplets(raw, pkg))
        assert out["certificate"]["resolve_delta_k"] <= E.RESOLVE_TOL_K
        assert out["optimized"]["peak"]["t_c"] <= out["initial"]["peak"]["t_c"] + 1e-9


class TestAssemblyKeepOut:
    """Chiplets that touch cannot be built: underfill has to flow between
    them and the microbump dispense has a tolerance. The rule is a minimum
    edge-to-edge clearance, and the optimizer has to respect it or it is
    returning layouts nobody can assemble."""

    @staticmethod
    def pairs_clear(placement, keepout_mm):
        """Every pair clears in x or in y, checked from the reported layout."""
        for a, b in itertools.combinations(placement, 2):
            gx = max(a["x_mm"], b["x_mm"]) - min(a["x_mm"] + a["w_mm"], b["x_mm"] + b["w_mm"])
            gy = max(a["y_mm"], b["y_mm"]) - min(a["y_mm"] + a["h_mm"], b["y_mm"] + b["h_mm"])
            if max(gx, gy) < keepout_mm - 1e-9:
                return False, (a["id"], b["id"], max(gx, gy))
        return True, None

    def test_the_mesh_rounds_the_keep_out_up_never_down(self):
        """A chiplet edge lands on a cell boundary, so a 100 um keep-out on a
        0.5 mm mesh is enforced as 0.5 mm. Rounding down would report a rule
        that was not applied."""
        assert spec(keepout_um=0.0).keepout_cells == 0
        fine = spec(size_mm=20.0, keepout_um=100.0)
        assert fine.cell_mm == 0.5
        assert (fine.keepout_cells, fine.keepout_enforced_um) == (1, 500.0)
        assert spec(size_mm=20.0, keepout_um=600.0).keepout_cells == 2
        assert spec(size_mm=30.0, keepout_um=100.0).keepout_enforced_um == 1000.0   # 1 mm mesh

    def test_touching_and_corner_touching_are_both_refused(self, cache):
        pkg = spec(keepout_um=100.0)
        touching = [chip("a", 2, 2, 3.0, 0, 0), chip("b", 2, 2, 3.0, 2, 0)]
        out = E.solve(cache, pkg, E.parse_chiplets(touching, pkg))
        assert out["ok"] is False and any("keep-out" in v for v in out["violations"])

        corner = [chip("a", 2, 2, 3.0, 0, 0), chip("b", 2, 2, 3.0, 2, 2)]
        out = E.solve(cache, pkg, E.parse_chiplets(corner, pkg))
        assert out["ok"] is False and any("keep-out" in v for v in out["violations"])

        # The same layouts are legal with the rule switched off.
        loose = spec(keepout_um=0.0)
        assert E.solve(cache, loose, E.parse_chiplets(touching, loose))["ok"]
        assert E.solve(cache, loose, E.parse_chiplets(corner, loose))["ok"]

    def test_the_optimizer_returns_a_layout_that_can_be_built(self, cache):
        pkg = spec(keepout_um=100.0)
        raw = [chip("hot", 2, 2, 6.0, 0, 0), chip("mid", 4, 2, 3.0, 4, 0), chip("big", 4, 4, 2.0, 0, 4)]
        out = E.optimize(cache, pkg, E.parse_chiplets(raw, pkg))
        ok, offender = self.pairs_clear(out["optimized"]["placement"], pkg.keepout_enforced_um * 1e-3)
        assert ok, f"optimizer returned an unbuildable layout: {offender}"
        cert = out["certificate"]
        assert cert["keepout_um"] == 100.0 and cert["keepout_enforced_um"] == 500.0

    def test_the_rule_costs_temperature_and_the_engine_says_so(self, cache):
        """The constrained optimum cannot be cooler than the unconstrained
        one -- and quoting the cheaper number would be quoting a layout that
        cannot be assembled."""
        raw = [chip("hot", 2, 2, 6.0, 0, 0), chip("mid", 4, 2, 3.0, 4, 0), chip("big", 4, 4, 2.0, 0, 4)]
        loose = spec(keepout_um=0.0)
        strict = spec(keepout_um=100.0)
        free = E.optimize(cache, loose, E.parse_chiplets(raw, loose))
        built = E.optimize(cache, strict, E.parse_chiplets(raw, strict))
        assert built["certificate"]["lattice_optimum_c"] >= free["certificate"]["lattice_optimum_c"] - 1e-9

    def test_a_start_that_cannot_be_built_is_refused_not_optimized(self, cache):
        pkg = spec(keepout_um=100.0)
        touching = E.parse_chiplets([chip("a", 2, 2, 3.0, 0, 0), chip("b", 2, 2, 3.0, 2, 0)], pkg)
        with pytest.raises(E.StudioError, match="keep-out"):
            E.optimize(cache, pkg, touching)


class TestStackEditor:
    """The stack is the customer's, not a constant: any whole number of 20 um
    cells per layer, with the mesh and the physics following."""

    def test_layers_and_mesh_follow_the_stack(self):
        default = spec()
        assert default.stack_um == E.STACK_UM
        assert default.nz == 12
        assert [l["name"] for l in default.layers] == list(E.LAYER_NAMES)
        assert [(l["k0"], l["k1"]) for l in default.layers] == [(0, 5), (5, 7), (7, 12)]

        taller = spec(stack_um=[200.0, 40.0, 100.0])
        assert taller.nz == 17
        assert taller.layers[-1]["k0"] == 12          # die starts above 240 um of stack
        assert taller.model_key != default.model_key  # so it is a different cached model

    def test_a_taller_stack_can_force_a_coarser_lateral_mesh(self):
        """20 mm at 0.5 mm cells is 1,600 columns; the direct solver path caps
        the total, so spending cells on z has to cost lateral resolution."""
        assert spec(size_mm=20.0).cell_mm == 0.5
        assert spec(size_mm=20.0, stack_um=[300.0, 60.0, 100.0]).cell_mm == 1.0

    @pytest.mark.parametrize("stack, message", [
        ([100.0, 30.0, 100.0], "whole number"),
        ([10.0, 40.0, 100.0], "stack"),
        ([400.0, 100.0, 200.0], "at most"),
        ([100.0, 40.0], "three thicknesses"),
        ("thick", "three thicknesses"),
    ])
    def test_an_unbuildable_stack_is_refused_with_a_reason(self, stack, message):
        with pytest.raises(E.StudioError, match=message):
            E.PackageSpec.parse({"size_mm": 10.0, "h_w_m2k": 2000.0, "ambient_c": 25.0,
                                 "pitch_mm": 2.0, "stack_um": stack})

    def test_a_thicker_interposer_spreads_heat_and_lowers_the_peak(self, cache):
        """The physical claim behind the control: more silicon under the
        chiplet is more room to spread. Compared at a FIXED mesh so the
        difference is the stack, not the discretization."""
        chiplet = [chip("hot", 2, 2, 6.0, 4, 4)]
        thin = spec(stack_um=[100.0, 40.0, 100.0])
        thick = spec(stack_um=[300.0, 40.0, 100.0])
        assert thin.cell_mm == thick.cell_mm            # same lateral mesh
        hot = E.solve(cache, thin, E.parse_chiplets(chiplet, thin))
        cool = E.solve(cache, thick, E.parse_chiplets(chiplet, thick))
        assert cool["peak"]["t_c"] < hot["peak"]["t_c"]
        # ... and both still conserve energy exactly.
        for out in (hot, cool):
            v = out["verification"]
            assert v["sink_heat_w"] == pytest.approx(v["specified_w"], rel=1e-9)

    def test_power_still_enters_at_the_active_plane_of_a_custom_stack(self, cache):
        pkg = spec(stack_um=[200.0, 60.0, 80.0])
        model, _ = cache.get(pkg)
        z0, z1 = model.power_z
        assert (z0, z1) == (13, 14)        # (200 + 60) / 20 = 13 cells below the die
        out = E.solve(cache, pkg, E.parse_chiplets([chip("a", 2, 2, 4.0, 2, 2)], pkg))
        assert out["peak"]["z_um"] == pytest.approx(270.0)     # first die cell centre
        assert out["peak"]["layer"] == "Die (Si)"


class TestMeshErrorBar:
    """The band is measured (tools/mesh_convergence.py), scaled to the design
    in hand, and absent rather than invented when no study has been run."""

    STUDY = {
        "cases": {
            "passive": {"order_p": 1.05, "band_pct_of_rise_by_cell_mm": {"0.5": 0.83, "1": 2.0, "2": 5.0}},
            "coldplate": {"order_p": 0.74, "band_pct_of_rise_by_cell_mm": {"0.5": 2.24, "1": 4.0, "2": 9.0}},
        }
    }

    def test_band_scales_with_the_rise_it_qualifies(self):
        small = E.mesh_band(0.5, 10.0, 1000.0, self.STUDY)
        large = E.mesh_band(0.5, 100.0, 1000.0, self.STUDY)
        assert small["band_k"] == pytest.approx(0.083)
        assert large["band_k"] == pytest.approx(0.83)
        assert small["band_pct_of_rise"] == large["band_pct_of_rise"] == 0.83

    def test_harder_cooling_uses_the_case_measured_for_it(self):
        assert E.mesh_band(0.5, 50.0, 1000.0, self.STUDY)["measured_on"] == "passive"
        assert E.mesh_band(0.5, 50.0, 25000.0, self.STUDY)["measured_on"] == "coldplate"

    def test_a_coarser_mesh_never_claims_a_tighter_band(self):
        fine = E.mesh_band(0.5, 50.0, 1000.0, self.STUDY)["band_k"]
        coarse = E.mesh_band(2.0, 50.0, 1000.0, self.STUDY)["band_k"]
        assert coarse > fine
        # An unmeasured cell size falls back to the coarsest measured one
        # rather than silently reporting the finest.
        unknown = E.mesh_band(4.0, 50.0, 1000.0, self.STUDY)
        assert unknown["cell_mm"] == 2.0

    def test_no_study_means_no_band_not_a_guess(self):
        assert E.mesh_band(0.5, 50.0, 1000.0, {}) is None
        assert E.mesh_band(0.5, 0.0, 1000.0, self.STUDY) is None

    def test_the_committed_study_is_readable_and_reaches_the_payload(self, cache):
        pkg = spec()
        chiplets = E.parse_chiplets([chip("a", 3, 3, 5.0, 0, 0)], pkg)
        out = E.solve(cache, pkg, chiplets)
        budget = out["metrics"]["uncertainty"]
        if budget is None:
            pytest.skip("no studies committed; run the tools in tools/")
        assert 0 < budget["combined_k"] < out["metrics"]["rise_k"]
        assert budget["cell_mm"] in (0.5, 1.0, 2.0)

    def test_the_budget_adds_independent_sources_in_quadrature(self):
        mesh = {"cases": {"passive": {"order_p": 1.0, "band_pct_of_rise_by_cell_mm": {"0.5": 1.0}}}}
        materials = {"cases": {"passive": {"material_band_k": 2.0, "rise_k": 50.0}}}
        budget = E.error_budget(0.5, 50.0, 1000.0, mesh, materials)
        assert budget["mesh_k"] == pytest.approx(0.5)      # 1 % of 50 K
        assert budget["material_k"] == pytest.approx(2.0)  # same rise as the study
        assert budget["combined_k"] == pytest.approx(math.hypot(0.5, 2.0))
        assert budget["combined_k"] < budget["mesh_k"] + budget["material_k"]

    def test_one_study_alone_still_gives_a_budget(self):
        mesh = {"cases": {"passive": {"band_pct_of_rise_by_cell_mm": {"0.5": 1.0}}}}
        only_mesh = E.error_budget(0.5, 50.0, 1000.0, mesh, {})
        assert only_mesh["material_k"] is None and only_mesh["combined_k"] == pytest.approx(0.5)
        assert E.error_budget(0.5, 50.0, 1000.0, {}, {}) is None


class TestExactOptimizer:
    def test_matches_independent_brute_force(self, cache):
        pkg = spec()
        raw = [chip("hot", 2, 2, 6.0, 0, 0), chip("mid", 4, 2, 3.0, 4, 0), chip("big", 4, 4, 2.0, 0, 4)]
        chiplets = E.parse_chiplets(raw, pkg)
        truth, _xy = brute_force_optimum(cache, pkg, raw)
        out = E.optimize(cache, pkg, chiplets)
        cert = out["certificate"]
        # Two different factorizations of the same system agree to ~1e-9 K.
        assert cert["lattice_optimum_c"] == pytest.approx(truth, abs=1e-7)
        assert out["optimized"]["peak"]["t_c"] <= out["initial"]["peak"]["t_c"] + 1e-9
        assert cert["resolve_delta_k"] <= E.RESOLVE_TOL_K

    def test_identical_chiplets_are_handled_exactly(self, cache):
        pkg = spec()
        # The start must itself satisfy the keep-out: the engine refuses to
        # optimize from a placement it would not report.
        raw = [chip("c0", 2, 2, 4.0, 0, 0), chip("c1", 2, 2, 4.0, 4, 0), chip("m", 4, 2, 2.0, 0, 6)]
        truth, _xy = brute_force_optimum(cache, pkg, raw)
        out = E.optimize(cache, pkg, E.parse_chiplets(raw, pkg))
        assert out["certificate"]["identical_chiplets"] == 1
        assert out["certificate"]["excluded_symmetry"] > 0
        # Two different factorizations of the same system agree to ~1e-9 K.
        assert out["certificate"]["lattice_optimum_c"] == pytest.approx(truth, abs=1e-7)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_bare_search_matches_brute_force_on_random_instances(self, cache, seed):
        """The branch-and-bound alone -- no starting layout, no greedy
        incumbent -- must find the brute-force optimum. Also checks the claim
        that searching die cells only is exact: brute force scores every cell."""
        rng = np.random.default_rng(seed)
        pkg = spec()
        raw = [chip(f"c{n}", float(rng.choice([2, 4])), float(rng.choice([2, 4])),
                    float(np.round(rng.uniform(1, 8), 2)), 0.0, 0.0) for n in range(3)]
        truth, _xy = brute_force_optimum(cache, pkg, raw)

        model, _ = cache.get(pkg)
        chiplets = E.parse_chiplets(raw, pkg)
        cell, pitch = pkg.cell_mm, int(round(pkg.pitch_mm / pkg.cell_mm))
        sizes = [(int(round(c.w_mm / cell)), int(round(c.h_mm / cell))) for c in chiplets]
        positions = [np.array([(x, y) for x in range(0, model.package.nx - w + 1, pitch)
                               for y in range(0, model.package.ny - h + 1, pitch)])
                     for w, h in sizes]
        with model.lock:
            fields = E.unit_fields(model, chiplets, positions, sizes)
        signatures = [(sizes[i], chiplets[i].power_w) for i in range(len(chiplets))]
        value, choice, stats = E.branch_and_bound(
            fields, positions, sizes, signatures, (np.inf, None),
            bound_cells=model.bound_cells, clearance_cells=pkg.keepout_cells,
            improve_incumbent=False)
        assert choice is not None
        assert stats.pruned_bound > 0          # the bound did real work
        # Two different factorizations of the same system agree to ~1e-9 K.
        assert value + pkg.ambient_c == pytest.approx(truth, abs=1e-7)

    def test_reproduces_the_projects_validated_optimum(self, cache):
        """The canonical experiment's optimum under every rule the product
        enforces: power at the active plane, the half-cell sink resistance,
        the 100 um assembly keep-out, and each chiplet's internal power map.
        With uniform chiplets and no keep-out the same case optimizes to
        70.2273 degC -- the difference is not noise, it is what those rules
        cost, and quoting the lower number would be quoting a package nobody
        can build out of parts nobody makes."""
        sc = next(s for s in scenarios() if s["id"] == "edge-ai-soc")
        pkg = E.PackageSpec.parse(sc["package"])
        out = E.optimize(cache, pkg, E.parse_chiplets(sc["chiplets"], pkg))
        assert out["certificate"]["lattice_optimum_c"] == pytest.approx(73.7656, abs=5e-5)
        assert out["certificate"]["combinations"] == 8_294_400

    def test_an_optimal_start_is_kept(self, cache):
        pkg = spec()
        out = E.optimize(cache, pkg, E.parse_chiplets([chip("solo", 2, 2, 5.0, 4, 4)], pkg))
        assert out["kept_initial"] is True
        assert out["improvement"]["delta_k"] == pytest.approx(0.0, abs=1e-9)

    def test_invalid_start_is_refused(self, cache):
        pkg = spec()
        chiplets = E.parse_chiplets([chip("a", 4, 4, 5.0, 0, 0), chip("b", 4, 4, 5.0, 2, 2)], pkg)
        with pytest.raises(E.StudioError, match="invalid"):
            E.optimize(cache, pkg, chiplets)

    def test_cancellation_stops_the_search(self, cache):
        pkg = spec()
        chiplets = E.parse_chiplets([chip("a", 2, 2, 5.0, 0, 0), chip("b", 2, 2, 3.0, 6, 6)], pkg)
        stop = threading.Event()
        stop.set()
        with pytest.raises(E.SearchCancelled):
            E.optimize(cache, pkg, chiplets, cancel=stop)

    def test_every_scenario_solves(self, cache):
        for sc in scenarios():
            pkg = E.PackageSpec.parse(sc["package"])
            out = E.solve(cache, pkg, E.parse_chiplets(sc["chiplets"], pkg))
            assert out["ok"], sc["id"]


@pytest.fixture(scope="module")
def base_url():
    server = serve("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


class TestHttpApi:
    @staticmethod
    def request(url, body=None, attempts=3):
        """One request, with a retry for a dropped connection ONLY.

        A refused connection or a reset socket is the local TCP stack, not
        the server's answer -- retrying it keeps the suite from failing for a
        reason that has nothing to do with the code. An HTTP status, however,
        IS the answer and is returned on the first try, never retried: a 404
        that the test expects must stay a 404.
        """
        data = None if body is None else json.dumps(body).encode("utf-8")
        for attempt in range(attempts):
            req = urllib.request.Request(url, data=data,
                                         headers={"Content-Type": "application/json"},
                                         method="POST" if body is not None else "GET")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return resp.status, resp.headers.get("Content-Type", ""), resp.read()
            except urllib.error.HTTPError as err:
                return err.code, err.headers.get("Content-Type", ""), err.read()
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if attempt == attempts - 1:
                    raise
                time.sleep(0.2)

    def test_app_and_assets_are_served(self, base_url):
        for path, kind in [("/", "text/html"), ("/css/app.css", "text/css")]:
            status, ctype, body = self.request(base_url + path)
            assert status == 200 and kind in ctype and body, path

    def test_every_imported_module_is_served(self, base_url):
        # A missing or misnamed ES module blanks the whole app, so follow the
        # import graph from app.js and fetch every module it reaches.
        import re
        pending, seen = ["app.js"], set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            status, ctype, body = self.request(f"{base_url}/js/{name}")
            assert status == 200 and "javascript" in ctype and body, name
            pending += re.findall(r'from\s+"\./([\w-]+\.js)"', body.decode("utf-8"))
        assert {"app.js", "viewer3d.js", "report.js", "mapdraw.js"} <= seen

    def test_paths_outside_the_app_are_refused(self, base_url):
        assert self.request(base_url + "/../run_studio.py")[0] == 404
        assert self.request(base_url + "/%2e%2e/run_studio.py")[0] == 404
        assert self.request(base_url + "/api/debug/snapshot", {"name": "x", "data_url": "x"})[0] == 404

    def test_solve_and_optimize_round_trip(self, base_url):
        status, _, body = self.request(base_url + "/api/scenarios")
        assert status == 200 and json.loads(body)["scenarios"]
        problem = {"package": {"size_mm": 10, "h_w_m2k": 2000, "ambient_c": 25, "pitch_mm": 2},
                   "chiplets": [chip("a", 2, 2, 5.0, 0, 0), chip("b", 4, 2, 3.0, 6, 6)]}
        status, _, body = self.request(base_url + "/api/solve", problem)
        solved = json.loads(body)
        assert status == 200 and solved["ok"] and solved["peak"]["t_c"] > 25.0

        status, _, body = self.request(base_url + "/api/optimize", problem)
        job_id = json.loads(body)["job_id"]
        deadline = time.time() + 120
        while True:
            job = json.loads(self.request(f"{base_url}/api/jobs/{job_id}")[2])["job"]
            if job["state"] != "running" or time.time() > deadline:
                break
            time.sleep(0.1)
        assert job["state"] == "done"
        result = job["result"]
        assert result["certificate"]["exact"] is True
        assert result["optimized"]["peak"]["t_c"] <= solved["peak"]["t_c"] + 1e-9

    def test_bad_requests_get_clear_errors(self, base_url):
        status, _, body = self.request(base_url + "/api/solve", {"package": {"size_mm": 10}})
        assert status == 400 and "h_w_m2k" in json.loads(body)["error"]
        assert self.request(base_url + "/api/jobs/nope")[0] == 404
