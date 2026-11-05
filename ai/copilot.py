"""
ai/copilot.py
ThermalForge :: state the goal in words; get constraints, exact runs and a cited
explanation (roadmap S4, step 4.4).

    "under 85 C at 30 W, memory on the left, SerDes at the edge"

becomes a plan -- a temperature limit, a power scaling, two placement
constraints -- that the exact optimizer runs; the explanation is built with
ai.records.Narrative, so every temperature in it is a citation of a solve
record and the text is audited before anyone sees it. The copilot is
structurally unable to state a temperature that is not in a solve record: its
prose has no way to carry one, and its output is refused if it does.

Reading the request is a TRAINED model (ai/language.py), not a list of
phrasings: it was trained on generated paraphrases and measured on hand-written
requests, and it abstains when it is not confident rather than applying the
nearest keyword match. The phrase grammar below is kept as the fallback for
when no model is registered, so the app still works with nothing trained.

Both are offline. The app runs with no network and the S4 rule is that it stays
that way, so the model is a few hundred kilobytes of weights that ship with the
repository, not a call to somebody's API. Writing the ANSWER stays templated and
cited; a language model can be put in front of that later (Explainer below), and
its text goes through the same audit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ai import language as lang
from ai import records as rec

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "memory": ("memory", "hbm", "dram", "sram", "mem", "cache"),
    "serdes": ("serdes", "io", "i/o", "phy", "transceiver"),
    "cpu": ("cpu", "cpus", "core", "cores", "processor", "risc", "risc-v", "rv"),
    "gpu": ("gpu", "graphics", "shader"),
    "npu": ("npu", "accelerator", "tpu", "ai"),
}

LADDER = (("direct", 2500.0), ("lid", 2500.0), ("lid_heatsink", 5000.0), ("coldplate", 25000.0))
NUM = r"(\d+(?:\.\d+)?)"


@dataclass
class Constraint:
    part_ids: List[str]
    kind: str                    # left right top bottom edge corner centre fixed
    words: str

    def describe(self, names: Dict[str, str]) -> str:
        who = ", ".join(names[i] for i in self.part_ids)
        return {"left": f"{who} in the left half", "right": f"{who} in the right half",
                "top": f"{who} in the top half", "bottom": f"{who} in the bottom half",
                "edge": f"{who} against a package edge", "corner": f"{who} in a corner",
                "centre": f"{who} in the central third", "fixed": f"{who} held where drawn"}[self.kind]


@dataclass
class Plan:
    text: str
    max_peak_c: Optional[float] = None
    total_power_w: Optional[float] = None
    part_power_w: Dict[str, float] = field(default_factory=dict)
    cooling: Optional[str] = None
    h_w_m2k: Optional[float] = None
    size_mm: Optional[float] = None
    constraints: List[Constraint] = field(default_factory=list)
    near: List[Tuple[str, str, str]] = field(default_factory=list)   # (a, b, words)
    understood: List[str] = field(default_factory=list)
    not_understood: List[str] = field(default_factory=list)


def _tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9/+-]+", s.lower())


def match_parts(phrase: str, chiplets: Sequence[dict]) -> List[str]:
    """Parts a phrase names: by id, by a word of the name, or by synonym."""
    words = set(_tokens(phrase))
    hits = []
    for c in chiplets:
        name_words = set(_tokens(c.get("name", ""))) | {c["id"].lower()}
        groups = {g for g, syn in SYNONYMS.items() if name_words & set(syn)}
        if words & name_words or any(words & set(SYNONYMS[g]) for g in groups):
            hits.append(c["id"])
    return hits


_MODEL: List[object] = []          # [] unknown, [None] none registered, [model]


def language_model():
    """The registered reader, or None. Loaded once; the copilot works either
    way, because the grammar is still there."""
    if not _MODEL:
        try:
            model, _card = lang.load_registered()
            _MODEL.append(model)
        except Exception:  # noqa: BLE001 -- nothing registered, or unreadable
            _MODEL.append(None)
    return _MODEL[0]


INTENT_KIND = {"LEFT": "left", "RIGHT": "right", "TOP": "top", "BOTTOM": "bottom",
               "EDGE": "edge", "CORNER": "corner", "CENTRE": "centre", "FIXED": "fixed"}
INTENT_COOLING = {"COOLING_DIRECT": "direct", "COOLING_LID": "lid",
                  "COOLING_HEATSINK": "lid_heatsink", "COOLING_COLDPLATE": "coldplate"}


def parse_learned(text: str, chiplets: Sequence[dict], model) -> Plan:
    """What the model read, turned into the plan the optimizer runs. The model
    decides what each clause ASKS for; the quantities are read deterministically,
    because there is nothing to learn about what 85 C means."""
    plan = Plan(text=text)
    for r in lang.read(text, chiplets, model):
        if not r.understood:
            plan.not_understood.append(r.text)
            continue
        got = lang.numbers(r.text)
        if r.intent == "LIMIT":
            value = got.get("temp", got.get("bare"))
            if value is None:
                plan.not_understood.append(r.text)
                continue
            plan.max_peak_c = value
        elif r.intent == "TOTAL_POWER":
            if "watt" not in got:
                plan.not_understood.append(r.text)
                continue
            plan.total_power_w = got["watt"]
        elif r.intent == "PART_POWER":
            if "watt" not in got:
                plan.not_understood.append(r.text)
                continue
            for pid in r.parts:
                plan.part_power_w[pid] = got["watt"]
        elif r.intent == "SIZE":
            value = got.get("mm", got.get("bare"))
            if value is None:
                plan.not_understood.append(r.text)
                continue
            plan.size_mm = value
        elif r.intent in INTENT_COOLING:
            plan.cooling = INTENT_COOLING[r.intent]
        elif r.intent == "NEAR":
            unique = list(dict.fromkeys(r.parts))
            plan.near.append((unique[0], unique[1], r.text))
        elif r.intent in INTENT_KIND:
            plan.constraints.append(Constraint(list(dict.fromkeys(r.parts)),
                                               INTENT_KIND[r.intent], r.text))
        elif r.intent != "OPTIMIZE":
            plan.not_understood.append(r.text)
            continue
        plan.understood.append(r.text)
    return plan


def parse(text: str, chiplets: Sequence[dict], model=None) -> Plan:
    """Read a request. With a registered model that is what reads it; without
    one, the phrase grammar below does, and says so by understanding less."""
    model = language_model() if model is None else model
    if model is not None:
        return parse_learned(text, chiplets, model)
    return parse_grammar(text, chiplets)


def parse_grammar(text: str, chiplets: Sequence[dict]) -> Plan:
    plan = Plan(text=text)
    clauses = [c.strip() for c in re.split(r"[,;]|\band\b|\bwith\b", text) if c.strip()]
    for clause in clauses:
        low = clause.lower()
        took = False
        m = re.search(r"(?:under|below|at most|no (?:more|hotter) than|max(?:imum)?(?: of)?|<=?|≤)\s*"
                      + NUM + r"\s*(?:°\s*c|degc|deg c|degrees|c\b)", low)
        if m:
            plan.max_peak_c = float(m.group(1))
            took = True
        who = match_parts(clause, chiplets)
        m = re.search(NUM + r"\s*w\b", low)
        if m and who and not re.search(r"\btotal\b", low):
            for pid in who:
                plan.part_power_w[pid] = float(m.group(1))
            took = True
        elif m:
            plan.total_power_w = float(m.group(1))
            took = True
        if re.search(r"cold ?plate|liquid", low):
            plan.cooling = "coldplate"; took = True
        elif re.search(r"heat ?sink", low):
            plan.cooling = "lid_heatsink"; took = True
        elif re.search(r"\blid\b", low):
            plan.cooling = "lid"; took = True
        elif re.search(r"\bbare\b|\bdirect\b", low):
            plan.cooling = "direct"; took = True
        m = re.search(r"\bh\s*=?\s*" + NUM, low)
        if m:
            plan.h_w_m2k = float(m.group(1)); took = True
        m = re.search(NUM + r"\s*mm\s*(?:x\s*\d+(?:\.\d+)?\s*mm\s*)?package|package\s*(?:of\s*)?" + NUM + r"\s*mm", low)
        if m:
            plan.size_mm = float(m.group(1) or m.group(2)); took = True
        if who:
            kind = None
            if re.search(r"\bfixed\b|where (?:it|they) (?:is|are)|don'?t move|do not move|keep .* in place", low):
                kind = "fixed"
            elif re.search(r"\bcorner", low):
                kind = "corner"
            elif re.search(r"\bedge|\bperimeter|\bboundary", low):
                kind = "edge"
            elif re.search(r"\bleft\b", low):
                kind = "left"
            elif re.search(r"\bright\b", low):
                kind = "right"
            elif re.search(r"\btop\b|\bnorth\b|\bupper\b", low):
                kind = "top"
            elif re.search(r"\bbottom\b|\bsouth\b|\blower\b", low):
                kind = "bottom"
            elif re.search(r"\bcent(?:re|er)|\bmiddle\b", low):
                kind = "centre"
            m = re.search(r"(.+?)\b(?:next to|near|beside|close to|adjacent to)\b(.+)", low)
            if m:
                a, b = match_parts(m.group(1), chiplets), match_parts(m.group(2), chiplets)
                if a and b and a[0] != b[0]:
                    plan.near.append((a[0], b[0], clause))
                    took = True
            elif kind:
                plan.constraints.append(Constraint(who, kind, clause))
                took = True
        if re.search(r"\b(?:optimi[sz]e|place|arrange|coolest|layout|floorplan)\b", low):
            took = True
        (plan.understood if took else plan.not_understood).append(clause)
    return plan


def allowed_fn(plan: Plan, chiplets: Sequence[dict], size_mm: float, size_y_mm: float,
               edge_mm: float) -> Optional[Callable[[int, float, float, float, float], bool]]:
    rules: Dict[int, List[str]] = {}
    index = {c["id"]: i for i, c in enumerate(chiplets)}
    for con in plan.constraints:
        if con.kind == "fixed":
            continue
        for pid in con.part_ids:
            rules.setdefault(index[pid], []).append(con.kind)
    if not rules:
        return None

    def ok(i: int, x: float, y: float, w: float, h: float) -> bool:
        cx, cy = x + w / 2, y + h / 2
        gaps = (x, y, size_mm - x - w, size_y_mm - y - h)
        for kind in rules.get(i, ()):
            if kind == "left" and cx > size_mm / 2 + 1e-9: return False
            if kind == "right" and cx < size_mm / 2 - 1e-9: return False
            if kind == "top" and cy < size_y_mm / 2 - 1e-9: return False
            if kind == "bottom" and cy > size_y_mm / 2 + 1e-9: return False
            if kind == "edge" and min(gaps) > edge_mm + 1e-9: return False
            if kind == "corner" and sorted(gaps)[1] > edge_mm + 1e-9: return False
            if kind == "centre" and not (size_mm / 3 <= cx <= 2 * size_mm / 3
                                         and size_y_mm / 3 <= cy <= 2 * size_y_mm / 3):
                return False
        return True
    return ok


def apply_plan(plan: Plan, package: dict, chiplets: List[dict]) -> Tuple[dict, List[dict]]:
    pkg = dict(package)
    if plan.size_mm is not None:
        pkg["size_mm"] = plan.size_mm
        pkg.pop("size_y_mm", None)
    if plan.cooling is not None:
        pkg["cooling"] = plan.cooling
        pkg.pop("cooling_layers", None)
        if plan.h_w_m2k is None:
            pkg["h_w_m2k"] = dict(LADDER)[plan.cooling]
    if plan.h_w_m2k is not None:
        pkg["h_w_m2k"] = plan.h_w_m2k
    parts = [dict(c) for c in chiplets]
    for c in parts:
        if c["id"] in plan.part_power_w:
            c["power_w"] = plan.part_power_w[c["id"]]
    if plan.total_power_w is not None:
        total = sum(c["power_w"] for c in parts)
        if total > 0:
            for c in parts:
                c["power_w"] = round(c["power_w"] * plan.total_power_w / total, 6)
    fixed = {pid for con in plan.constraints if con.kind == "fixed" for pid in con.part_ids}
    for c in parts:
        if c["id"] in fixed:
            c["fixed"] = True
    return pkg, parts


def near_links(plan: Plan, chiplets: Sequence[dict]) -> Tuple[List[Tuple[str, str, float]], Optional[float]]:
    if not plan.near:
        return [], None
    by_id = {c["id"]: c for c in chiplets}
    links, cap = [], 0.0
    for a, b, _words in plan.near:
        ca, cb = by_id[a], by_id[b]
        links.append((a, b, 1.0))
        # "near": centres no further apart (Manhattan) than side by side plus 2 mm
        cap += (ca["w_mm"] + cb["w_mm"]) / 2 + (ca["h_mm"] + cb["h_mm"]) / 2 + 2.0
    return links, cap


@dataclass
class Answer:
    plan: Plan
    narrative: str
    records: List[str]
    placement: Optional[List[dict]]
    package: dict
    meets_limit: Optional[bool]

    def as_dict(self) -> dict:
        p = self.plan
        return {"understood": p.understood, "not_understood": p.not_understood,
                "plan": {"max_peak_c": p.max_peak_c, "total_power_w": p.total_power_w,
                         "part_power_w": p.part_power_w, "cooling": p.cooling, "h_w_m2k": p.h_w_m2k,
                         "size_mm": p.size_mm,
                         "constraints": [{"parts": c.part_ids, "kind": c.kind} for c in p.constraints],
                         "near": [[a, b] for a, b, _ in p.near]},
                "narrative": self.narrative, "display": rec.strip_markers(self.narrative),
                "citations": [{"record": r, "path": path} for _k, r, path in rec.iter_citations(self.narrative)],
                "records": self.records, "placement": self.placement, "package": self.package,
                "meets_limit": self.meets_limit}


class Explainer:
    """Turns results into words. The default is templates; anything else --
    a language model, say -- is audited, and falls back to the template if
    it states a temperature it cannot cite."""

    def explain(self, store: rec.RecordStore, ctx: dict) -> str:  # pragma: no cover - interface
        raise NotImplementedError


def _template(store: rec.RecordStore, ctx: dict) -> str:
    n = rec.Narrative(store)
    plan: Plan = ctx["plan"]
    names = ctx["names"]
    start: rec.SolveRecord = ctx["start"]
    runs: List[Tuple[str, rec.SolveRecord]] = ctx["runs"]
    request: Optional[rec.SolveRecord] = ctx.get("request")
    n.text("As drawn, the design peaks at ").cite(rec.cite(start, "peak_c")).text(".")
    if plan.constraints or plan.near:
        cons = [c.describe(names) for c in plan.constraints]
        cons += [f"{names[a]} near {names[b]}" for a, b, _ in plan.near]
        n.text(" Constraints applied exactly: " + "; ".join(cons) + ".")
    label, first = runs[0]
    n.text(f" The proven coolest layout{' under those constraints' if plan.constraints or plan.near else ''}"
           f" with {label} peaks at ").cite(rec.cite(first, "optimized.peak_c")).text(".")
    if request is not None and plan.max_peak_c is not None:
        n.text(" The limit asked for is ").cite(rec.cite(request, "max_peak_c")).text(".")
        met = [(lbl, r) for lbl, r in runs if r.value("optimized.peak_c") <= plan.max_peak_c]
        if met and met[0][1] is first:
            n.text(" That meets it, with ").cite(
                rec.cite(ctx["margins"][first.id], "value", unit="K")).text(" to spare.")
        elif met:
            lbl, r = met[0]
            n.text(f" It does not, so stronger cooling was tried: with {lbl} the proven optimum is ")
            n.cite(rec.cite(r, "optimized.peak_c")).text(", which meets it with ")
            n.cite(rec.cite(ctx["margins"][r.id], "value", unit="K")).text(" to spare.")
        else:
            for lbl, r in runs[1:]:
                n.text(f" With {lbl}: ").cite(rec.cite(r, "optimized.peak_c")).text(".")
            lbl, r = min(runs, key=lambda pair: pair[1].value("optimized.peak_c"))
            n.text(f" Nothing tried meets it; the closest, {lbl}, is ").cite(
                rec.cite(ctx["margins"][r.id], "value", unit="K")).text(" over.")
    if plan.not_understood:
        n.text(" Not understood, so not applied: "
               + "; ".join(f"'{rec.redact(w)}'" for w in plan.not_understood) + ".")
    return n.render()


def answer(text: str, package: dict, chiplets: List[dict], store: rec.RecordStore, cache,
           explainer: Optional[Explainer] = None, try_cooling: bool = True,
           max_extra_runs: int = 2, model=None) -> Answer:
    import studio.engine as E
    plan = parse(text, chiplets, model=model)
    pkg, parts = apply_plan(plan, package, chiplets)
    names = {c["id"]: c.get("name", c["id"]) for c in parts}
    spec = E.PackageSpec.parse(pkg)
    edge_mm = max(spec.pitch_mm, 1.0)
    allow = allowed_fn(plan, parts, spec.size_mm, spec.size_y, edge_mm)
    links, cap = near_links(plan, parts)
    notes = [c.describe(names) for c in plan.constraints] + [f"{names[a]} near {names[b]}"
                                                            for a, b, _ in plan.near]
    start = rec.record_solve(store, cache, pkg, parts)
    request = rec.record_request(store, text, {"max_peak_c": plan.max_peak_c}) \
        if plan.max_peak_c is not None else None

    def run(p: dict, label: str) -> Tuple[str, rec.SolveRecord]:
        r = rec.record_optimize(store, cache, p, parts, allowed=allow, search="guided",
                                links=links or None, wirelength_cap_mm=cap, constraint_notes=notes)
        return label, r

    cooling_label = lambda p: f"{p.get('cooling', 'direct').replace('_', ' + ')} cooling (h {p['h_w_m2k']:g})"
    try:
        runs = [run(pkg, cooling_label(pkg))]
    except E.StudioError as exc:
        n = rec.Narrative(store)
        n.text("As drawn, the design peaks at ").cite(rec.cite(start, "peak_c")).text(
            ". No layout was proven, because " + rec.redact(str(exc)) + ".")
        if plan.not_understood:
            n.text(" Not understood, so not applied: "
                   + "; ".join(f"'{rec.redact(w)}'" for w in plan.not_understood) + ".")
        return Answer(plan=plan, narrative=n.render(), records=[start.id], placement=None,
                      package=pkg, meets_limit=None)
    if (try_cooling and plan.max_peak_c is not None and plan.cooling is None
            and runs[0][1].value("optimized.peak_c") > plan.max_peak_c):
        names_ladder = [c for c, _ in LADDER]
        now = pkg.get("cooling", "direct")
        rung = names_ladder.index(now) if now in names_ladder else -1
        for cooling, h in LADDER[rung + 1:rung + 1 + max_extra_runs]:
            alt = dict(pkg, cooling=cooling, h_w_m2k=max(h, float(pkg["h_w_m2k"])))
            alt.pop("cooling_layers", None)
            runs.append(run(alt, cooling_label(alt)))
            if runs[-1][1].value("optimized.peak_c") <= plan.max_peak_c:
                break
    margins = {}
    if request is not None:
        for _lbl, r in runs:
            pair = {"limit": (request, "max_peak_c"), "peak": (r, "optimized.peak_c")}
            over = r.value("optimized.peak_c") > plan.max_peak_c
            margins[r.id] = rec.record_derived(store, pair, "peak - limit" if over else "limit - peak")
    ctx = {"plan": plan, "names": names, "start": start, "runs": runs, "request": request,
           "margins": margins}
    narrative = None
    if explainer is not None:
        try:
            candidate = explainer.explain(store, ctx)
            rec.audit(candidate, store)
            narrative = candidate
        except rec.CitationError:
            narrative = None            # refused: fall back to the template
    if narrative is None:
        narrative = _template(store, ctx)
    chosen = next((r for _l, r in runs if plan.max_peak_c is not None
                   and r.value("optimized.peak_c") <= plan.max_peak_c), runs[0][1])
    meets = None if plan.max_peak_c is None else chosen.value("optimized.peak_c") <= plan.max_peak_c
    ids = [start.id] + ([request.id] if request else []) + [r.id for _l, r in runs] + \
        [m.id for m in margins.values()]
    return Answer(plan=plan, narrative=narrative, records=ids, placement=chosen.value("placement"),
                  package=chosen.inputs["package"], meets_limit=meets)
