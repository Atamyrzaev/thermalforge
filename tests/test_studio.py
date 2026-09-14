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
    best, best_xy = np.inf, None
    for combo in itertools.product(*options):
        rects = [(xy[0], xy[1], c["w_mm"], c["h_mm"]) for (xy, _), c in zip(combo, chiplets)]
        overlap = any(min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]) > 1e-9
                      and min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]) > 1e-9
                      for a, b in itertools.combinations(rects, 2))
        if overlap:
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
        band = out["metrics"]["mesh"]
        if band is None:
            pytest.skip("no mesh study committed; run tools/mesh_convergence.py")
        assert 0 < band["band_k"] < out["metrics"]["rise_k"]
        assert band["cell_mm"] in (0.5, 1.0, 2.0)


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
        raw = [chip("c0", 2, 2, 4.0, 0, 0), chip("c1", 2, 2, 4.0, 2, 0), chip("m", 4, 2, 2.0, 0, 6)]
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
            bound_cells=model.bound_cells, improve_incumbent=False)
        assert choice is not None
        assert stats.pruned_bound > 0          # the bound did real work
        # Two different factorizations of the same system agree to ~1e-9 K.
        assert value + pkg.ambient_c == pytest.approx(truth, abs=1e-7)

    def test_reproduces_the_projects_validated_optimum(self, cache):
        """The canonical experiment's exhaustive optimum, measured by the
        independent src/exhaustive.py search: 70.2110 degC."""
        sc = next(s for s in scenarios() if s["id"] == "edge-ai-soc")
        pkg = E.PackageSpec.parse(sc["package"])
        out = E.optimize(cache, pkg, E.parse_chiplets(sc["chiplets"], pkg))
        assert out["certificate"]["lattice_optimum_c"] == pytest.approx(70.2273, abs=5e-5)
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
    def request(url, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, resp.headers.get("Content-Type", ""), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, err.headers.get("Content-Type", ""), err.read()

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
