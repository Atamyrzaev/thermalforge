"""
tests/test_ai.py
RISC3D-AI :: the AI layer (roadmap S4), held to its one rule.

AI proposes, the solver decides. These tests pin that rule in the places it
could be broken: a temperature that no solve backs, a dataset that changed
after a model was measured on it, a search that returns a different optimum
because something "guided" it, a constraint that is enforced loosely, a
calibration that reports two confident numbers where the data only determines
one. Every module here is allowed to be wrong about SPEED. None of them is
allowed to be wrong about what a number means.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import studio.engine as E
from ai import calibration as CAL
from ai import copilot as CO
from ai import dataset as D
from ai import guided as G
from ai import inverse as INV
from ai import records as REC
from ai import registry as REG
from ai import rom as ROM
from ai import surrogate as SUR
from studio.scenarios import scenarios


@pytest.fixture(scope="module")
def cache():
    return E.ModelCache(capacity=2)


@pytest.fixture(scope="module")
def demo():
    sc = scenarios()[0]
    return sc["package"], [dict(c) for c in sc["chiplets"]]


def small_design(size=10.0, parts=2, pattern="mac_array"):
    pkg = {"size_mm": size, "h_w_m2k": 2500.0, "ambient_c": 25.0, "pitch_mm": 2.0}
    chips = [{"id": f"p{i}", "name": f"P{i}", "w_mm": 3.0, "h_mm": 3.0, "power_w": 4.0 + i,
              "power_pattern": pattern if i == 0 else "uniform",
              "x_mm": 0.0 + 5 * i, "y_mm": 0.0} for i in range(parts)]
    return pkg, chips


# ---------------------------------------------------------------------------
# records: the only way a temperature reaches a user
# ---------------------------------------------------------------------------

class TestSolveRecords:
    def test_a_record_exists_only_for_a_solve_that_passed(self, cache):
        store = REC.RecordStore()
        pkg, chips = small_design()
        r = REC.record_solve(store, cache, pkg, chips)
        assert r.kind == "solve" and r.checks["passed"]
        assert r.value("peak_c") > pkg["ambient_c"]
        assert store.get(r.id) is r and len(store) == 1

    def test_the_same_solve_is_the_same_record(self, cache):
        """The id is a hash of inputs, outputs and checks: citing a number
        twice cites one record, and a record cannot be quietly replaced."""
        store = REC.RecordStore()
        pkg, chips = small_design()
        a = REC.record_solve(store, cache, pkg, chips)
        b = REC.record_solve(store, cache, pkg, chips)
        assert a.id == b.id and len(store) == 1

    def test_prose_may_not_state_a_temperature(self):
        store = REC.RecordStore()
        n = REC.Narrative(store)
        for bad in ("it runs at 85 C", "about 85.4 °C", "a rise of 12 K", "85degC"):
            with pytest.raises(REC.CitationError):
                n.text(bad)
        n.text("it runs cool at 30 W over 4 parts")          # numbers without a unit are fine

    def test_the_audit_catches_an_uncited_or_forged_number(self, cache):
        store = REC.RecordStore()
        pkg, chips = small_design()
        r = REC.record_solve(store, cache, pkg, chips)
        good = REC.Narrative(store).text("the peak is ").cite(REC.cite(r, "peak_c")).render()
        assert REC.audit(good, store)
        with pytest.raises(REC.CitationError):
            REC.audit("the peak is 42.0 °C", store)                      # uncited
        with pytest.raises(REC.CitationError):
            REC.audit(f"the peak is 1.0 °C [rec:{r.id}:peak_c]", store)   # cited, wrong value
        with pytest.raises(REC.CitationError):
            REC.audit("the peak is 42.0 °C [rec:000000000000:peak_c]", store)  # no such record

    def test_a_citation_must_point_at_a_number_in_the_record(self, cache):
        store = REC.RecordStore()
        pkg, chips = small_design()
        r = REC.record_solve(store, cache, pkg, chips)
        with pytest.raises(REC.CitationError):
            REC.cite(r, "mesh")                      # a string, not a number
        with pytest.raises(KeyError):
            REC.cite(r, "no_such_field")

    def test_a_margin_is_a_record_too(self, cache):
        """Arithmetic on cited numbers is still traceable: the derived record
        names its parents, so 'with 7 K to spare' can be checked."""
        store = REC.RecordStore()
        pkg, chips = small_design()
        r = REC.record_solve(store, cache, pkg, chips)
        req = REC.record_request(store, "under 200 C", {"max_peak_c": 200.0})
        d = REC.record_derived(store, {"limit": (req, "max_peak_c"), "peak": (r, "peak_c")},
                               "limit - peak")
        assert d.value("value") == pytest.approx(200.0 - r.value("peak_c"))
        assert d.inputs["parents"]["peak"]["id"] == r.id

    def test_user_text_echoed_back_has_its_temperatures_redacted(self):
        assert "85" not in REC.redact("keep it under 85 C please")

    def test_records_survive_a_round_trip(self, cache, tmp_path):
        path = tmp_path / "records.jsonl"
        store = REC.RecordStore(path)
        pkg, chips = small_design()
        r = REC.record_solve(store, cache, pkg, chips)
        again = REC.RecordStore(path)
        assert again.get(r.id).value("peak_c") == r.value("peak_c")


# ---------------------------------------------------------------------------
# dataset and registry: a measured claim names exactly what it was measured on
# ---------------------------------------------------------------------------

class TestDatasetAndRegistry:
    def test_a_dataset_that_changed_is_refused(self, tmp_path):
        result = D.generate(2, 2, seed=5, workers=1, escalate=False)
        gen = {"packages": 2, "layouts": 2, "seed": 5, "mesh": "standard"}
        D.write("tiny", 1, result, gen, root=tmp_path)
        assert D.load("tiny", 1, root=tmp_path).manifest["counts"]["labels"] == len(result["labels"])
        path = tmp_path / "tiny" / "v1" / "labels.jsonl.gz"
        path.write_bytes(path.read_bytes() + b"\x00")
        with pytest.raises(D.DatasetError, match="sha256"):
            D.load("tiny", 1, root=tmp_path)

    def test_a_version_is_immutable(self, tmp_path):
        result = D.generate(1, 1, seed=6, workers=1, escalate=False)
        gen = {"packages": 1, "layouts": 1, "seed": 6, "mesh": "standard"}
        D.write("tiny", 1, result, gen, root=tmp_path)
        with pytest.raises(D.DatasetError, match="immutable"):
            D.write("tiny", 1, result, gen, root=tmp_path)

    def test_the_split_is_by_package_and_covers_every_package(self):
        split = D.split_packages(20, seed=3)
        ids = split["train"] + split["calib"] + split["test"]
        assert sorted(ids) == list(range(20))
        assert not (set(split["train"]) & set(split["test"]))

    def test_every_label_carries_its_checks_and_its_band(self):
        result = D.generate(1, 2, seed=7, workers=1, escalate=False)
        for lab in result["labels"]:
            assert lab["checks"]["passed"] and lab["band_k"] > 0
            assert lab["clean"] is not lab["peak_is_lower_bound"]

    def test_a_model_that_changed_is_refused(self, tmp_path):
        version, path = REG.register("toy", {"w": [1.0]}, {"metrics": {}}, root=tmp_path)
        params, card = REG.load("toy", version, root=tmp_path)
        assert params["w"] == [1.0] and card["version"] == version
        (path / "model.json").write_text('{"w": [2.0]}', encoding="utf-8")
        with pytest.raises(REG.RegistryError, match="sha256"):
            REG.load("toy", version, root=tmp_path)

    def test_every_run_is_logged_kept_or_not(self, tmp_path):
        log = tmp_path / "experiments.jsonl"
        REG.log_run("train", {"a": 1}, {"err": 2.0}, path=log)
        REG.log_run("train", {"a": 2}, {"err": 9.0}, path=log)
        runs = REG.runs(path=log)
        assert len(runs) == 2 and runs[0]["config"]["a"] == 1


# ---------------------------------------------------------------------------
# the reduced-order model: physics, cheaply
# ---------------------------------------------------------------------------

class TestReducedOrderModel:
    def test_power_is_conserved_however_coarse_the_cell(self):
        pkg, chips = small_design()
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        for cell in (0.5, 1.0, 2.0):
            nx = int(round(spec.size_mm / cell))
            q = ROM.part_power_map(parts[0], nx, nx, cell)
            assert q.sum() == pytest.approx(parts[0].power_w, rel=1e-9)

    def test_it_agrees_with_the_engine_on_the_same_design(self, cache):
        """Not exact -- it collapses each layer to one node -- but close
        enough that what the learned correction has to fix is small."""
        pkg, chips = small_design(size=20.0, parts=3)
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        truth = E.solve(cache, spec, parts)
        sol = ROM.solve_layout(ROM.RomCache(factor=1.0), spec, parts)
        rise = truth["peak"]["t_c"] - spec.ambient_c
        assert max(sol.part_rise) == pytest.approx(rise, rel=0.15)

    def test_a_colder_sink_gives_a_colder_package(self):
        pkg, chips = small_design()
        cache_ = ROM.RomCache(factor=2.0)
        rises = []
        for h in (1000.0, 25000.0):
            spec = E.PackageSpec.parse(dict(pkg, h_w_m2k=h))
            parts = E.parse_chiplets(chips, spec)
            rises.append(max(ROM.solve_layout(cache_, spec, parts).part_rise))
        assert rises[0] > rises[1] * 2

    def test_the_stack_follows_the_engine_spec(self):
        spec = E.PackageSpec.parse({"size_mm": 10.0, "h_w_m2k": 2500.0, "ambient_c": 25.0,
                                    "pitch_mm": 1.0, "cooling": "lid_heatsink"})
        names = [p.name for p in ROM.plates(spec)]
        assert names[:3] == ["interposer", "bumps", "device"]
        assert "Copper lid" in names and "Heatsink base" in names


# ---------------------------------------------------------------------------
# the surrogate: an estimate that says so
# ---------------------------------------------------------------------------

class TestSurrogate:
    @staticmethod
    def toy_model(rom_factor=2.0):
        rng = np.random.default_rng(0)
        n = len(SUR.FEATURES)
        members = [[rng.normal(0, 0.01, (n, 4)), np.zeros(4), rng.normal(0, 0.01, (4, 4)),
                    np.zeros(4), rng.normal(0, 0.01, (4, 1)), np.zeros(1)]]
        return SUR.Surrogate(mean=np.zeros(n), std=np.ones(n), members=members,
                             rom_factor=rom_factor, q95_rel=0.1)

    def test_a_prediction_is_labelled_an_estimate_and_carries_an_interval(self):
        pkg, chips = small_design()
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        out = self.toy_model().predict(ROM.RomCache(factor=2.0), spec, parts)
        assert out["label"] == "surrogate estimate"
        lo, hi = out["interval_c"]
        assert lo < out["peak_c"] < hi

    def test_features_are_finite_and_named(self):
        pkg, chips = small_design(parts=3)
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        sol = ROM.solve_layout(ROM.RomCache(factor=2.0), spec, parts)
        feats = SUR.part_features(spec, parts, sol)
        assert feats.shape == (3, len(SUR.FEATURES))
        assert np.isfinite(feats).all()

    def test_a_model_trained_on_other_features_is_refused(self):
        blob = self.toy_model().to_json()
        blob["features"] = blob["features"][:-1]
        with pytest.raises(ValueError, match="feature set"):
            SUR.Surrogate.from_json(blob)

    def test_it_survives_a_round_trip(self):
        model = self.toy_model()
        again = SUR.Surrogate.from_json(json.loads(json.dumps(model.to_json())))
        pkg, chips = small_design()
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        a = model.predict(ROM.RomCache(factor=2.0), spec, parts)["peak_c"]
        b = again.predict(ROM.RomCache(factor=2.0), spec, parts)["peak_c"]
        assert a == pytest.approx(b, rel=1e-12)

    def test_what_if_scores_every_position_it_is_given(self):
        pkg, chips = small_design(size=20.0, parts=2)
        spec = E.PackageSpec.parse(pkg)
        parts = E.parse_chiplets(chips, spec)
        spots = [(0.0, 0.0), (10.0, 10.0), (16.0, 16.0)]
        out = SUR.what_if(ROM.RomCache(factor=2.0), spec, parts, 0, spots, model=self.toy_model())
        assert [p["x_mm"] for p in out["positions"]] == [0.0, 10.0, 16.0]
        assert out["best"]["peak_c"] == min(p["peak_c"] for p in out["positions"])
        assert "not a reported temperature" in out["label"]


# ---------------------------------------------------------------------------
# the guided search: faster, and the same answer
# ---------------------------------------------------------------------------

def capture_search(cache, spec, chiplets):
    """The arrays the engine hands its search, so a policy can be compared
    against the shipped one on exactly the same problem."""
    grabbed = {}
    real = E.branch_and_bound

    def spy(fields, positions, extents, signatures, incumbent, **kw):
        grabbed.update(fields=fields, positions=positions, extents=extents,
                       signatures=signatures, incumbent=incumbent, kw=kw)
        return real(fields, positions, extents, signatures, incumbent, **kw)

    E.branch_and_bound = spy
    try:
        out = E.optimize(cache, spec, chiplets, search="shipped")
    finally:
        E.branch_and_bound = real
    return out, grabbed


class TestGuidedSearch:
    @pytest.fixture(scope="class")
    def problem(self):
        """One instance, searched once by the engine. Single-threaded on
        purpose: the shipped search shares its incumbent between threads, so
        how many layouts it scores depends on which thread got there first,
        and a tree can only be compared with a tree that was not raced."""
        threads = E.MAX_SEARCH_THREADS
        E.MAX_SEARCH_THREADS = 1
        try:
            cache = E.ModelCache(capacity=1)
            pkg = {"size_mm": 20.0, "h_w_m2k": 2500.0, "ambient_c": 25.0, "pitch_mm": 4.0}
            chips = [{"id": f"c{i}", "name": f"C{i}", "w_mm": 4.0, "h_mm": 4.0,
                      "power_w": 2.0 + i, "power_pattern": "mac_array" if i % 2 else "uniform",
                      "x_mm": 6.0 * (i % 3), "y_mm": 6.0 * (i // 3)} for i in range(4)]
            spec = E.PackageSpec.parse(pkg)
            parts = E.parse_chiplets(chips, spec)
            out, grabbed = capture_search(cache, spec, parts)
            yield spec, parts, out, grabbed, cache
        finally:
            E.MAX_SEARCH_THREADS = threads

    def run(self, grabbed, **kw):
        c = grabbed
        return G.guided_branch_and_bound(
            c["fields"], c["positions"], c["extents"], c["signatures"], c["incumbent"],
            bound_cells=c["kw"].get("bound_cells"),
            clearance_cells=c["kw"].get("clearance_cells", 0), **kw)

    def test_the_shipped_policy_reproduces_the_shipped_tree(self, problem):
        _spec, _parts, out, grabbed, _cache = problem
        value, _choice, stats = self.run(grabbed)
        cert = out["certificate"]
        assert value == pytest.approx(cert["lattice_optimum_c"] - _spec.ambient_c, abs=1e-9)
        assert stats.leaves == cert["layouts_scored"]

    def test_the_screen_changes_the_work_and_not_the_answer(self, problem):
        _spec, _parts, _out, grabbed, _cache = problem
        plain_v, plain_c, plain_s = self.run(grabbed)
        screened_v, screened_c, screened_s = self.run(grabbed, prescreen=True, prior_cells=4)
        assert screened_v == pytest.approx(plain_v, abs=1e-12)
        assert screened_c == plain_c
        assert screened_s.leaves <= plain_s.leaves
        assert screened_s.leaves + screened_s.prescreened == plain_s.leaves

    def test_every_policy_returns_the_same_optimum(self, problem):
        _spec, _parts, _out, grabbed, _cache = problem
        base, _c, _s = self.run(grabbed)
        for policy in (G.shipped_policy, G.fail_first_policy):
            value, _choice, _stats = self.run(grabbed, policy=policy, prescreen=True)
            assert value == pytest.approx(base, abs=1e-9)

    def test_a_policy_that_names_an_ineligible_part_is_refused(self, problem):
        _spec, _parts, _out, grabbed, _cache = problem
        with pytest.raises(ValueError, match="not eligible"):
            self.run(grabbed, policy=lambda view: 999)

    def test_the_engine_option_agrees_with_the_shipped_search(self, problem):
        """The guided search is what the app runs by default; the shipped one
        stays as the baseline it is measured against."""
        spec, parts, out, _grabbed, cache = problem
        import inspect
        assert inspect.signature(E.optimize).parameters["search"].default == "guided"
        guided = E.optimize(cache, spec, parts)          # the default
        assert guided["certificate"]["lattice_optimum_c"] == pytest.approx(
            out["certificate"]["lattice_optimum_c"], abs=1e-9)
        assert guided["certificate"]["search"] == "guided"
        assert guided["certificate"]["layouts_scored"] <= out["certificate"]["layouts_scored"]

    def test_a_wirelength_cap_is_exact(self, problem):
        """Capped, the search may only return a layout inside the cap -- and
        its peak cannot be better than the uncapped optimum."""
        spec, parts, out, _grabbed, cache = problem
        links = [(parts[0].id, parts[1].id, 1.0)]
        free = E.optimize(cache, spec, parts, search="guided", links=links)["certificate"]
        cap = free["wirelength_mm"] * 0.8
        capped = E.optimize(cache, spec, parts, search="guided", links=links,
                            wirelength_cap_mm=cap)["certificate"]
        assert capped["wirelength_mm"] <= cap + 1e-9
        assert capped["lattice_optimum_c"] >= free["lattice_optimum_c"] - 1e-9
        assert capped["constrained"] is True

    def test_a_cap_nothing_can_meet_is_refused_rather_than_guessed(self, problem):
        spec, parts, _out, _grabbed, cache = problem
        links = [(parts[0].id, parts[1].id, 1.0)]
        with pytest.raises(E.StudioError, match="constraints"):
            E.optimize(cache, spec, parts, search="guided", links=links, wirelength_cap_mm=0.5)

    def test_a_wirelength_cap_needs_the_guided_search(self, problem):
        """The threaded search has no side-constraint pruning, so asking it
        for one is refused rather than quietly ignored."""
        spec, parts, _out, _grabbed, cache = problem
        with pytest.raises(E.StudioError, match="guided"):
            E.optimize(cache, spec, parts, search="shipped",
                       links=[(parts[0].id, parts[1].id, 1.0)], wirelength_cap_mm=10.0)

    def test_a_region_constraint_is_enforced_exactly(self, problem):
        spec, parts, out, _grabbed, cache = problem
        left = E.optimize(cache, spec, parts, search="guided",
                          allowed=lambda i, x, y, w, h: i != 0 or x + w / 2 <= spec.size_mm / 2,
                          constraint_notes=["C0 in the left half"])
        placed = left["optimized"]["placement"][0]
        assert placed["x_mm"] + placed["w_mm"] / 2 <= spec.size_mm / 2 + 1e-9
        assert left["certificate"]["constrained"] is True
        assert left["certificate"]["constraints"] == ["C0 in the left half"]
        # constraining can only cost temperature, never win it
        assert (left["certificate"]["lattice_optimum_c"]
                >= out["certificate"]["lattice_optimum_c"] - 1e-9)


# ---------------------------------------------------------------------------
# the copilot: words in, constraints and cited answers out
# ---------------------------------------------------------------------------

class TestCopilotParsing:
    PARTS = [{"id": "cpu0", "name": "CPU Core", "w_mm": 4, "h_mm": 4, "power_w": 3.0,
              "x_mm": 0, "y_mm": 0},
             {"id": "hbm0", "name": "HBM Memory", "w_mm": 6, "h_mm": 6, "power_w": 2.0,
              "x_mm": 8, "y_mm": 0},
             {"id": "ser0", "name": "SerDes", "w_mm": 2, "h_mm": 4, "power_w": 1.0,
              "x_mm": 0, "y_mm": 10}]

    def test_it_reads_a_limit_a_power_and_two_regions(self):
        plan = CO.parse("under 85 C at 30 W, memory on the left, SerDes at the edge", self.PARTS)
        assert plan.max_peak_c == 85.0 and plan.total_power_w == 30.0
        kinds = {c.kind: c.part_ids for c in plan.constraints}
        assert kinds["left"] == ["hbm0"] and kinds["edge"] == ["ser0"]
        assert not plan.not_understood

    def test_it_says_what_it_did_not_understand(self):
        plan = CO.parse("make it faster and prettier", self.PARTS)
        assert plan.not_understood and not plan.constraints

    @pytest.mark.parametrize("phrase,kind", [
        ("CPU in the corner", "corner"), ("CPU in the centre", "centre"),
        ("CPU at the bottom", "bottom"), ("CPU on the right", "right"),
        ("keep the CPU where it is", "fixed")])
    def test_placement_words(self, phrase, kind):
        plan = CO.parse(phrase, self.PARTS)
        assert [c.kind for c in plan.constraints] == [kind]

    def test_synonyms_find_the_part(self):
        assert CO.match_parts("the dram", self.PARTS) == ["hbm0"]
        assert CO.match_parts("i/o", self.PARTS) == ["ser0"]

    def test_cooling_and_size(self):
        plan = CO.parse("use a cold plate on a 30 mm package", self.PARTS)
        assert plan.cooling == "coldplate" and plan.size_mm == 30.0

    def test_near_becomes_a_link_with_a_cap(self):
        plan = CO.parse("put the CPU next to the memory", self.PARTS)
        assert plan.near and plan.near[0][:2] == ("cpu0", "hbm0")
        links, cap = CO.near_links(plan, self.PARTS)
        assert links == [("cpu0", "hbm0", 1.0)] and cap > 0

    def test_a_region_rule_keeps_only_the_positions_it_allows(self):
        plan = CO.parse("memory on the left", self.PARTS)
        ok = CO.allowed_fn(plan, self.PARTS, 20.0, 20.0, 1.0)
        assert ok(1, 0.0, 0.0, 6.0, 6.0) and not ok(1, 14.0, 0.0, 6.0, 6.0)
        assert ok(0, 14.0, 0.0, 4.0, 4.0)        # the rule applies to memory only

    def test_power_is_scaled_and_parts_are_fixed(self):
        plan = CO.parse("at 12 W, keep the SerDes where it is", self.PARTS)
        pkg, parts = CO.apply_plan(plan, {"size_mm": 20.0}, self.PARTS)
        assert sum(p["power_w"] for p in parts) == pytest.approx(12.0)
        assert next(p for p in parts if p["id"] == "ser0")["fixed"] is True


class TestCopilotAnswers:
    def test_every_temperature_in_an_answer_is_cited(self, cache, demo):
        pkg, chips = demo
        store = REC.RecordStore()
        answer = CO.answer("under 120 C, memory on the left", pkg, chips, store, cache)
        cites = REC.audit(answer.narrative, store)
        assert cites and answer.meets_limit is True
        for _kind, rid, path in REC.iter_citations(answer.narrative):
            assert store.get(rid).value(path) is not None

    def test_an_explainer_that_invents_a_temperature_is_refused(self, cache, demo):
        class Liar(CO.Explainer):
            def explain(self, store, ctx):
                return "It will run at 42.0 °C, trust me."

        pkg, chips = demo
        store = REC.RecordStore()
        answer = CO.answer("under 120 C", pkg, chips, store, cache, explainer=Liar())
        assert "42.0" not in answer.narrative
        REC.audit(answer.narrative, store)

    def test_an_explainer_that_misquotes_a_record_is_refused(self, cache, demo):
        class Forger(CO.Explainer):
            def explain(self, store, ctx):
                return f"The optimum is 1.0 °C [rec:{ctx['runs'][0][1].id}:optimized.peak_c]."

        pkg, chips = demo
        store = REC.RecordStore()
        answer = CO.answer("under 120 C", pkg, chips, store, cache, explainer=Forger())
        assert "1.0 °C" not in answer.narrative

    def test_constraints_that_cannot_be_met_are_said_so(self, cache, demo):
        pkg, chips = demo
        store = REC.RecordStore()
        answer = CO.answer("memory on the left, NPU in the centre, CPU in the centre",
                           pkg, chips, store, cache)
        assert answer.placement is None
        assert "no layout was proven" in answer.narrative.lower()
        REC.audit(answer.narrative, store)

    def test_the_copilot_needs_no_network(self):
        source = (pathlib.Path(CO.__file__)).read_text(encoding="utf-8")
        for module in ("import socket", "import urllib", "import requests", "http.client"):
            assert module not in source


# ---------------------------------------------------------------------------
# inverse design: proposals are screened, but reported numbers are proven
# ---------------------------------------------------------------------------

class TestInverseDesign:
    def test_a_candidate_must_be_able_to_hold_the_parts(self):
        parts = [{"id": "a", "name": "A", "w_mm": 8.0, "h_mm": 8.0, "power_w": 5.0,
                  "x_mm": 0, "y_mm": 0}]
        sizes = {c.size_mm for c in INV.candidates(parts, {"ambient_c": 25.0})}
        assert 10.0 not in sizes and 20.0 in sizes

    def test_the_pareto_front_keeps_what_nothing_dominates(self):
        pts = [(10.0, 1.0), (9.0, 2.0), (11.0, 3.0), (9.5, 1.5)]
        assert INV.pareto(pts) == [True, True, False, True]

    def test_a_proposal_is_an_exact_optimum_with_a_record(self, monkeypatch, cache):
        monkeypatch.setattr(INV, "SIZES_MM", (10.0, 15.0))
        monkeypatch.setattr(INV, "COOLINGS", (("direct", 2500.0), ("coldplate", 25000.0)))
        monkeypatch.setattr(INV, "INTERPOSERS_UM", (100.0,))
        parts = [{"id": "a", "name": "A", "w_mm": 3.0, "h_mm": 3.0, "power_w": 6.0,
                  "power_pattern": "mac_array", "x_mm": 0.0, "y_mm": 0.0},
                 {"id": "b", "name": "B", "w_mm": 3.0, "h_mm": 3.0, "power_w": 3.0,
                  "x_mm": 5.0, "y_mm": 0.0}]
        store = REC.RecordStore()
        out = INV.design(parts, {"ambient_c": 25.0}, store, cache, max_peak_c=200.0,
                         n_small=1, n_best=1)
        assert out["proposal"] is not None
        record = store.get(out["proposal"]["record"])
        assert record.kind == "optimize" and record.checks["exact"]
        assert out["proposal"]["peak_c"] == pytest.approx(record.value("optimized.peak_c"))
        assert out["exact_runs"] <= 2 < out["screened"]


# ---------------------------------------------------------------------------
# the calibration learner: what the data determines, and what it does not
# ---------------------------------------------------------------------------

class TestCalibrationLearner:
    @staticmethod
    def synthetic(h_factor: float):
        """Measurements made by the engine itself at a known h, so the fit has
        a right answer to find."""
        pkg = {"size_mm": 8.0, "h_w_m2k": 15000.0, "ambient_c": 25.0, "pitch_mm": 1.0,
               "cooling": "lid", "keepout_um": 0.0, "stack_um": [20.0, 20.0, 100.0]}
        chips = [{"id": f"h{n}", "name": f"H{n}", "w_mm": 2.0, "h_mm": 2.0,
                  "power_w": 2.0 if n in (0, 5) else 0.2,
                  "x_mm": (n % 4) * 2.0, "y_mm": (n // 4) * 2.0} for n in range(16)]
        sensors = [((n % 4) * 2.0 + 1.0, (n // 4) * 2.0 + 1.0) for n in range(16)]
        case = CAL.MeasuredCase("synthetic", pkg, chips, sensors, [0.0] * 16)
        case.measured_c = list(CAL.forward(case, {"h": math.log(h_factor)}))
        return case

    def test_it_finds_a_known_coefficient(self):
        case = self.synthetic(1.4)
        result = CAL.fit([case], params=("h",), max_iter=8)
        assert result.values["h"] == pytest.approx(15000.0 * 1.4, rel=0.05)
        assert result.rms_k < 0.05
        assert result.sigma_factor["h"] > 1.0

    def test_what_the_data_can_separate_is_reported_as_separable(self):
        """Sixteen sensors across the die do separate h from the TIM: the TIM
        sits under a spreading lid, so it changes the SHAPE as well as the
        level. The fit must not cry degeneracy when there is none."""
        case = self.synthetic(1.2)
        result = CAL.fit([case], params=("h", "k_tim"), max_iter=4)
        assert not result.not_identifiable
        assert abs(result.correlation[0][1]) < 0.95

    def test_two_parameters_one_sensor_cannot_separate_are_flagged(self):
        """With a single sensor there is only a level to fit, and h and a
        series interface resistance both move it: the fit has to say the pair
        is not determined rather than report two confident numbers."""
        case = self.synthetic(1.2)
        case.sensors, case.measured_c = case.sensors[:1], case.measured_c[:1]
        result = CAL.fit([case], params=("h", "k_tim"), max_iter=4)
        assert result.not_identifiable, result.correlation

    def test_the_family_reader_builds_one_case_per_measured_pattern(self):
        cases = CAL.ttc_family()
        assert len(cases) == 13
        for c in cases:
            assert len(c.sensors) == len(c.measured_c) == 16
            assert c.package["cooling"] == "lid_heatsink"
