"""
ai/records.py
RISC3D-AI :: solve records -- the only thing a temperature may be quoted from.

The S4 rule "no number reaches a user without a passing solve behind it" has
to live in code, not in a prompt. This is that code.

    SolveRecord   one verified solve (or optimization): its inputs, its
                  outputs, its checks, the commit that produced it, and an id
                  that is a hash of all of that. Nothing else can mint one;
                  record_solve() and record_optimize() run the real engine.
    Citation      (record id, field path) -> the value stored there. The only
                  way a temperature gets into text.
    Narrative     a text builder with no method that accepts a bare
                  temperature. Prose goes in through .text(), which REFUSES
                  anything that looks like one; numbers go in through
                  .cite(), which reads them from a record.
    audit()       the second wall: any temperature-like token in the final
                  text must sit inside a citation marker whose record exists
                  and whose stored value rounds to the printed one. Text that
                  fails is not shown -- it raises.

Two walls, because either alone has a hole: the builder cannot stop a
template author from typing "85 C" into a format string, and the audit alone
would let a caller cite a record that was never produced. Together, a
temperature in the output is a pointer into a solve that passed its checks.

User-stated limits ("under 85 C") are quoted through a request record, so the
text can restate the question without anything pretending it was computed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

# A temperature: a number followed by a temperature unit. Kelvin differences
# ("+1.3 K") count too -- a band or a delta is a claim about temperature.
_TEMP = re.compile(r"(?<![\w.])([-+]?\d+(?:\.\d+)?)\s*(?:°\s*C|° ?F|degC|deg C|K\b|C\b)")
# Citation marker as rendered: "85.76 °C [rec:1a2b3c4d5e6f:peak_c]"
_MARK = re.compile(r"\[(rec|req):([0-9a-f]{12}):([A-Za-z0-9_.\[\]-]+)\]")


class CitationError(Exception):
    """Text states a temperature that no record backs."""


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best-effort outside a checkout
        return "unknown"


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)


def _lookup(data: dict, path: str) -> Any:
    """'chiplet_peaks_c.cpu' or 'moves[0].max_after_c'."""
    cur: Any = data
    for token in re.findall(r"[^.\[\]]+|\[\d+\]", path):
        if token.startswith("["):
            cur = cur[int(token[1:-1])]
        else:
            cur = cur[token]
    return cur


@dataclass(frozen=True)
class SolveRecord:
    id: str
    kind: str                        # "solve", "optimize" or "request"
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    checks: Dict[str, Any]
    commit: str
    created: float

    def value(self, path: str) -> Any:
        return _lookup(self.outputs, path)

    def as_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "inputs": self.inputs, "outputs": self.outputs,
                "checks": self.checks, "commit": self.commit, "created": self.created}


def _mint(kind: str, inputs: dict, outputs: dict, checks: dict) -> SolveRecord:
    commit = git_commit()
    digest = hashlib.sha256(_canon({"kind": kind, "inputs": inputs, "outputs": outputs,
                                    "checks": checks, "commit": commit}).encode()).hexdigest()
    return SolveRecord(id=digest[:12], kind=kind, inputs=inputs, outputs=outputs,
                       checks=checks, commit=commit, created=time.time())


class RecordStore:
    """Records by id, optionally appended to a JSONL file so a session's
    citations survive it. Thread-safe: the server calls it from handlers."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._by_id: Dict[str, SolveRecord] = {}
        self._lock = threading.Lock()
        self.path = path
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self._by_id[d["id"]] = SolveRecord(**d)

    def put(self, record: SolveRecord) -> SolveRecord:
        with self._lock:
            if record.id not in self._by_id:
                self._by_id[record.id] = record
                if self.path is not None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record.as_dict()) + "\n")
        return self._by_id[record.id]

    def get(self, record_id: str) -> SolveRecord:
        with self._lock:
            if record_id not in self._by_id:
                raise CitationError(f"no solve record {record_id}")
            return self._by_id[record_id]

    def __contains__(self, record_id: str) -> bool:
        return record_id in self._by_id

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> List[SolveRecord]:
        with self._lock:
            return list(self._by_id.values())


# ---------------------------------------------------------------------------
# Minting records from the real engine -- the only way one comes to exist
# ---------------------------------------------------------------------------

def _solve_outputs(out: dict) -> Tuple[dict, dict]:
    u = out["metrics"]["uncertainty"] or {}
    outputs = {
        "peak_c": float(out["peak"]["t_c"]),
        "rise_k": float(out["metrics"]["rise_k"]),
        "ambient_c": float(out["ambient_c"]),
        "band_k": u.get("combined_k"),
        "peak_is_lower_bound": bool(u.get("peak_is_lower_bound")),
        "chiplet_peaks_c": {r["id"]: float(r["max_c"]) for r in out["chiplets"]},
        "hottest_chiplet": out.get("hottest_chiplet"),
        "total_power_w": float(out["metrics"]["total_power_w"]),
        "mesh": out["grid"]["mesh"], "cell_mm": float(out["grid"].get("cell_mm", 0.0) or 0.0),
    }
    checks = {"residual": float(out["verification"]["residual"]),
              "energy_imbalance_k": float(out["verification"]["energy_imbalance_k"]),
              "passed": True}
    return outputs, checks


def record_solve(store: RecordStore, cache, package: dict, chiplets: List[dict]) -> SolveRecord:
    """Run the verified solver and keep what it said. The engine raises if a
    per-solve check fails, so a record only exists for a solve that passed."""
    import studio.engine as E
    spec = E.PackageSpec.parse(package)
    out = E.solve(cache, spec, E.parse_chiplets(chiplets, spec))
    if not out.get("ok"):
        raise CitationError("the placement is invalid and was not solved: "
                            + "; ".join(out.get("violations", [])))
    outputs, checks = _solve_outputs(out)
    return store.put(_mint("solve", {"package": package, "chiplets": chiplets}, outputs, checks))


def record_optimize(store: RecordStore, cache, package: dict, chiplets: List[dict],
                    **optimize_kw) -> SolveRecord:
    import studio.engine as E
    spec = E.PackageSpec.parse(package)
    out = E.optimize(cache, spec, E.parse_chiplets(chiplets, spec), **optimize_kw)
    before, _ = _solve_outputs(out["initial"])
    after, checks = _solve_outputs(out["optimized"])
    cert = out["certificate"]
    outputs = {"initial": before, "optimized": after,
               "delta_k": float(out["improvement"]["delta_k"]),
               "placement": out["optimized"]["placement"],
               "moves": [{k: m[k] for k in ("id", "name", "from_mm", "to_mm", "max_before_c",
                                            "max_after_c")} for m in out["moves"]],
               "certificate": {k: cert.get(k) for k in (
                   "exact", "combinations", "layouts_scored", "layouts_prescreened", "search",
                   "constrained", "constraints", "wirelength_mm", "wirelength_cap_mm",
                   "lattice_optimum_c", "resolve_delta_k", "total_s")}}
    checks = {**checks, "resolve_delta_k": float(cert["resolve_delta_k"]), "exact": True}
    # the options that define the problem, in a form that hashes the same in
    # every session (a constraint FUNCTION would hash its memory address)
    options = {"search": optimize_kw.get("search", "shipped"),
               "links": [list(l) for l in (optimize_kw.get("links") or [])],
               "wirelength_cap_mm": optimize_kw.get("wirelength_cap_mm"),
               "constraints": list(optimize_kw.get("constraint_notes") or [])}
    return store.put(_mint("optimize", {"package": package, "chiplets": chiplets,
                                        "options": options}, outputs, checks))


def record_derived(store: RecordStore, parents: Dict[str, Tuple[SolveRecord, str]],
                   rule: str) -> SolveRecord:
    """Arithmetic on cited values, kept as a record of its own: the inputs name
    the parent records and fields, the rule is evaluated here, and the result
    can be cited like anything else. Supported rules are differences only --
    'a - b' -- because a margin is the only derived number the copilot states."""
    a_name, b_name = [t.strip() for t in rule.split("-")]
    a_rec, a_path = parents[a_name]
    b_rec, b_path = parents[b_name]
    for rec in (a_rec, b_rec):
        if rec.id not in store:
            raise CitationError(f"parent record {rec.id} is not in this store")
    value = float(a_rec.value(a_path)) - float(b_rec.value(b_path))
    inputs = {"rule": rule, "parents": {n: {"id": r.id, "path": p} for n, (r, p) in parents.items()}}
    return store.put(_mint("derived", inputs, {"value": value}, {"passed": True}))


def record_request(store: RecordStore, text: str, limits: Dict[str, float]) -> SolveRecord:
    """What the user asked for, so a stated limit can be quoted -- as a
    request, never as something computed."""
    return store.put(_mint("request", {"text": text}, dict(limits), {"passed": True}))


# ---------------------------------------------------------------------------
# Citations, the narrative builder, and the audit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Citation:
    record: SolveRecord
    path: str
    unit: str = "°C"                # "°C" for temperatures, "K" for differences
    digits: int = 1

    @property
    def value(self) -> float:
        return float(self.record.value(self.path))

    def render(self) -> str:
        prefix = "req" if self.record.kind == "request" else "rec"
        return f"{self.value:.{self.digits}f} {self.unit} [{prefix}:{self.record.id}:{self.path}]"


def cite(record: SolveRecord, path: str, unit: str = "°C", digits: int = 1) -> Citation:
    value = record.value(path)          # KeyError if the record does not hold it
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CitationError(f"{record.id}:{path} is not a number")
    return Citation(record, path, unit, digits)


class Narrative:
    """Text with temperatures that can only come from records."""

    def __init__(self, store: RecordStore) -> None:
        self.store = store
        self._parts: List[str] = []
        self.cited: List[Citation] = []

    def text(self, prose: str) -> "Narrative":
        hit = _TEMP.search(prose)
        if hit:
            raise CitationError(f"prose may not state a temperature ({hit.group(0)!r}); "
                                f"cite a record instead")
        self._parts.append(prose)
        return self

    def cite(self, citation: Citation) -> "Narrative":
        if citation.record.id not in self.store:
            raise CitationError(f"record {citation.record.id} is not in this store")
        self.cited.append(citation)
        self._parts.append(citation.render())
        return self

    def render(self) -> str:
        out = "".join(self._parts)
        audit(out, self.store)
        return out


def audit(text: str, store: RecordStore) -> List[Tuple[str, str, float]]:
    """Every temperature-like token must be a rendered citation that matches
    its record. Returns the citations found; raises CitationError otherwise."""
    found = []
    for m in _TEMP.finditer(text):
        tail = text[m.end():m.end() + 64]
        mark = _MARK.match(tail.lstrip())
        if not mark:
            raise CitationError(f"uncited temperature {m.group(0)!r} in: "
                                f"...{text[max(0, m.start() - 40):m.end() + 10]}...")
        _kind, rid, path = mark.groups()
        record = store.get(rid)
        stored = float(record.value(path))
        printed = float(m.group(1))
        decimals = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
        if abs(abs(stored) - abs(printed)) > 0.5 * 10 ** (-decimals) + 1e-9:
            raise CitationError(f"{m.group(0)!r} does not match record {rid}:{path} = {stored}")
        found.append((rid, path, stored))
    return found


def redact(text: str) -> str:
    """User text echoed back: its temperatures are not the model's to state."""
    return _TEMP.sub("[a temperature]", text)


def strip_markers(text: str) -> str:
    """For display: the audited text without its markers (the UI links them)."""
    return re.sub(r" ?" + _MARK.pattern, "", text).replace("  ", " ")


def records_cited(text: str) -> List[str]:
    return sorted({m.group(2) for m in _MARK.finditer(text)})


def iter_citations(text: str) -> Iterable[Tuple[str, str, str]]:
    for m in _MARK.finditer(text):
        yield m.groups()
