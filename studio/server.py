"""
studio/server.py
RISC3D Studio :: local web server and JSON API.

Standard library only, so the demo runs offline on any machine with the
project's Python environment: no framework, no CDN, no build step.

    GET  /                      the app
    GET  /api/scenarios         built-in demo scenarios
    POST /api/solve             solve one placement (fast: cached factorization)
    POST /api/optimize          start an exact optimization job -> {job_id}
    POST /api/transient         start a time-stepped burst -> {job_id}
    GET  /api/jobs/<id>         job progress and, when done, the result
    POST /api/jobs/<id>/cancel  stop a running job
    POST /api/jobs/<id>/pause   hold it at the next safe point
    POST /api/jobs/<id>/resume  let it carry on
"""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional

from ai import records as ai_records
from ai import rom as ai_rom
from studio import engine
from studio.scenarios import scenarios

STATIC_ROOT = Path(__file__).resolve().parent / "static"
QA_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "studio_qa"
MAX_BODY_BYTES = 1_000_000
MAX_JOBS = 16

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


class Control:
    """Cancellation, and pause, in the object the engine already consults.

    The search and the time stepper call `cancel.is_set()` at every point
    where stopping is safe. That makes those same points the only places a
    pause can safely take effect, so pausing is implemented by BLOCKING in
    `is_set()` rather than by threading a second flag through the engine: no
    solver code changes, and a paused job cannot be holding a half-updated
    incumbent. Cancelling releases the hold, so a paused job can still be
    stopped.
    """

    def __init__(self):
        self._cancel = threading.Event()
        self._go = threading.Event()
        self._go.set()
        self._held = threading.Event()
        self.paused = False

    # -- what the engine sees --------------------------------------------
    def is_set(self) -> bool:
        if not self._go.is_set():
            # Asking for a pause is not the same as being paused. The engine
            # only consults this between safe points, and model building has
            # none, so a request can sit unhonoured for seconds. Saying
            # "paused" during that window would be a claim the job has stopped
            # when it has not.
            self._held.set()
            try:
                self._go.wait()      # released by resume() or by set()
            finally:
                self._held.clear()
        return self._cancel.is_set()

    @property
    def held(self) -> bool:
        return self._held.is_set()

    def set(self) -> None:           # cancel
        self._cancel.set()
        self._go.set()               # never leave a thread parked on a dead job

    # -- what the server sees --------------------------------------------
    def pause(self) -> None:
        if not self._cancel.is_set():
            self.paused = True
            self._go.clear()

    def resume(self) -> None:
        self.paused = False
        self._go.set()


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self.state = "running"
        self.progress: Dict = {"stage": "queued", "elapsed_s": 0.0}
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.cancel = Control()
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            state = self.state
            if state == "running" and self.cancel.paused:
                state = "paused" if self.cancel.held else "pausing"
            out = {"id": self.id, "state": state, "progress": dict(self.progress)}
            if self.state == "done":
                out["result"] = self.result
            if self.error:
                out["error"] = self.error
            return out


class Studio:
    """Shared state: the model cache, the optimization jobs, and the AI layer
    (solve records, the reduced-order model, the registered surrogate)."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.cache = engine.ModelCache()
        self.jobs: Dict[str, Job] = {}
        self.jobs_lock = threading.Lock()
        self.records = ai_records.RecordStore()
        self.rom = ai_rom.RomCache(capacity=8)
        self._surrogate = None
        self._surrogate_card = None
        self._surrogate_tried = False

    @property
    def surrogate(self):
        """The registered surrogate, or None when none is registered. Loaded
        once, lazily: the app runs, and every number it reports comes from the
        solver, whether or not a model is present."""
        if not self._surrogate_tried:
            self._surrogate_tried = True
            try:
                from ai import surrogate as ai_surrogate
                self._surrogate, self._surrogate_card = ai_surrogate.load_registered()
                self.rom = ai_rom.RomCache(capacity=8, factor=self._surrogate.rom_factor)
            except Exception:  # noqa: BLE001 -- no model registered, or an unreadable one
                self._surrogate, self._surrogate_card = None, None
        return self._surrogate

    def _start(self, name: str, work) -> Job:
        """Runs one long job on its own thread, reporting progress and
        honouring a stop. Shared by the optimizer and the time stepper: both
        take seconds, both must be cancellable, and neither may take the
        server down with it."""
        job = Job()
        with self.jobs_lock:
            if len(self.jobs) >= MAX_JOBS:
                for old in sorted(self.jobs.values(), key=lambda j: j.created):
                    if old.state != "running":
                        del self.jobs[old.id]
                        break
            self.jobs[job.id] = job

        def progress(update: dict) -> None:
            with job.lock:
                job.progress = update

        def run() -> None:
            try:
                result = work(progress, job.cancel)
                with job.lock:
                    job.result, job.state = result, "done"
            except engine.SearchCancelled:
                with job.lock:
                    job.state, job.error = "cancelled", f"{name} stopped"
            except engine.StudioError as exc:
                with job.lock:
                    job.state, job.error = "error", str(exc)
            except Exception as exc:  # noqa: BLE001 - reported to the UI, logged here
                traceback.print_exc()
                with job.lock:
                    job.state, job.error = "error", f"internal error: {exc}"

        threading.Thread(target=run, name=f"{name}-{job.id}", daemon=True).start()
        return job

    def start_optimization(self, spec, chiplets) -> Job:
        return self._start("optimization", lambda progress, cancel: engine.optimize(
            self.cache, spec, chiplets, on_progress=progress, cancel=cancel))

    def start_transient(self, spec, chiplets, profile) -> Job:
        return self._start("transient", lambda progress, cancel: engine.transient(
            self.cache, spec, chiplets, profile, on_progress=progress, cancel=cancel))

    def start_inverse(self, parts, package, max_peak_c, links) -> Job:
        from ai import inverse as ai_inverse

        def work(progress, cancel):
            progress({"stage": "screening candidates", "elapsed_s": 0.0})
            return ai_inverse.design(parts, package, self.records, self.cache,
                                     max_peak_c=max_peak_c, links=links,
                                     surrogate=self.surrogate)

        return self._start("inverse", work)


def _frange(start: float, stop: float, step: float):
    n = int((stop - start) / step + 1e-9)
    return [start + i * step for i in range(max(0, n) + 1)]


def make_handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        server_version = "RISC3DStudio/1.0"

        def log_message(self, fmt, *args):  # quiet: API polling would flood the console
            if not self.path.startswith("/api/jobs/"):
                super().log_message(fmt, *args)

        # -- helpers -----------------------------------------------------
        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _error(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
            self._send_json({"ok": False, "error": message}, status)

        def _read_json(self, limit: int = MAX_BODY_BYTES) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > limit:
                raise engine.StudioError("request body missing or too large")
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise engine.StudioError(f"request is not valid JSON: {exc}") from exc
            if not isinstance(data, dict):
                raise engine.StudioError("request must be a JSON object")
            return data

        def _parse_problem(self):
            data = self._read_json()
            spec = engine.PackageSpec.parse(data.get("package"))
            chiplets = engine.parse_chiplets(data.get("chiplets"), spec)
            return spec, chiplets

        def _serve_static(self, rel: str) -> None:
            target = (STATIC_ROOT / rel).resolve()
            if STATIC_ROOT not in target.parents and target != STATIC_ROOT:
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                self._error("not found", HTTPStatus.NOT_FOUND)
                return
            body = target.read_bytes()
            ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype == "application/javascript":
                ctype += "; charset=utf-8"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        # -- routes ------------------------------------------------------
        def do_GET(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/scenarios":
                    # The internal power patterns travel with the scenarios so
                    # the UI never keeps its own copy of the physics.
                    self._send_json({
                        "ok": True,
                        "scenarios": scenarios(),
                        "power_patterns": {
                            name: [list(block) for block in blocks]
                            for name, blocks in engine.POWER_PATTERNS.items()
                        },
                    })
                elif path.startswith("/api/records/"):
                    try:
                        record = studio.records.get(path.rsplit("/", 1)[-1])
                    except ai_records.CitationError as exc:
                        self._error(str(exc), HTTPStatus.NOT_FOUND)
                    else:
                        self._send_json({"ok": True, "record": record.as_dict()})
                elif path == "/api/ai/status":
                    model = studio.surrogate
                    card = studio._surrogate_card or {}
                    self._send_json({"ok": True, "surrogate": None if model is None else {
                        "model": f"{card.get('name')}/v{card.get('version')}",
                        "dataset": card.get("dataset", {}).get("id"),
                        "p95_err_over_band": card.get("gate", {}).get("p95_err_over_band"),
                        "interval_pct_of_rise": None if model.q95_rel is None else model.q95_rel * 100,
                        "note": "estimates only; every reported temperature comes from a solve"},
                        "records": len(studio.records)})
                elif path.startswith("/api/jobs/"):
                    job = studio.jobs.get(path.rsplit("/", 1)[-1])
                    if job is None:
                        self._error("no such job", HTTPStatus.NOT_FOUND)
                    else:
                        self._send_json({"ok": True, "job": job.snapshot()})
                elif path.startswith("/api/"):
                    self._error("not found", HTTPStatus.NOT_FOUND)
                else:
                    self._serve_static(path.lstrip("/") or "index.html")
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._error(f"internal error: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

        def do_POST(self):  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if path == "/api/solve":
                    spec, chiplets = self._parse_problem()
                    self._send_json(engine.solve(studio.cache, spec, chiplets))
                elif path == "/api/optimize":
                    spec, chiplets = self._parse_problem()
                    job = studio.start_optimization(spec, chiplets)
                    self._send_json({"ok": True, "job_id": job.id})
                elif path == "/api/mesh-check":
                    # Seconds, not milliseconds: it solves the design twice.
                    spec, chiplets = self._parse_problem()
                    self._send_json(engine.mesh_check(studio.cache, spec, chiplets))
                elif path == "/api/transient":
                    data = self._read_json()
                    spec = engine.PackageSpec.parse(data.get("package"))
                    chiplets = engine.parse_chiplets(data.get("chiplets"), spec)
                    profile = engine.parse_profile(data.get("transient"))
                    job = studio.start_transient(spec, chiplets, profile)
                    self._send_json({"ok": True, "job_id": job.id})
                elif path == "/api/copilot":
                    from ai import copilot as ai_copilot
                    data = self._read_json()
                    text = str(data.get("text", ""))[:500]
                    if not text.strip():
                        raise engine.StudioError("ask something")
                    answer = ai_copilot.answer(text, data.get("package"), data.get("chiplets"),
                                               studio.records, studio.cache)
                    self._send_json({"ok": True, **answer.as_dict()})
                elif path == "/api/surrogate/what-if":
                    from ai import surrogate as ai_surrogate
                    data = self._read_json()
                    spec = engine.PackageSpec.parse(data.get("package"))
                    chiplets = engine.parse_chiplets(data.get("chiplets"), spec)
                    part_id = str(data.get("part_id", ""))
                    index = next((i for i, c in enumerate(chiplets) if c.id == part_id), None)
                    if index is None:
                        raise engine.StudioError("what-if needs the id of a part in this design")
                    moved = chiplets[index]
                    gap = spec.keepout_enforced_um / 1000.0

                    def clear(x, y):
                        return not any(x < o.x_mm + o.w_mm + gap and o.x_mm < x + moved.w_mm + gap
                                       and y < o.y_mm + o.h_mm + gap and o.y_mm < y + moved.h_mm + gap
                                       for i, o in enumerate(chiplets) if i != index)

                    spots = [(x, y) for x in _frange(0.0, spec.size_mm - moved.w_mm, spec.pitch_mm)
                             for y in _frange(0.0, spec.size_y - moved.h_mm, spec.pitch_mm)
                             if clear(x, y)]
                    out = ai_surrogate.what_if(studio.rom, spec, chiplets, index, spots,
                                               model=studio.surrogate)
                    self._send_json({"ok": True, "part_id": part_id, **out})
                elif path == "/api/inverse":
                    data = self._read_json()
                    spec = engine.PackageSpec.parse(data.get("package"))
                    engine.parse_chiplets(data.get("chiplets"), spec)     # validate
                    links = [tuple(l) for l in (data.get("links") or [])]
                    limit = data.get("max_peak_c")
                    job = studio.start_inverse(list(data.get("chiplets")), data.get("package"),
                                               float(limit) if limit is not None else None, links)
                    self._send_json({"ok": True, "job_id": job.id})
                elif path == "/api/debug/snapshot" and studio.debug:
                    # Developer aid (python run_studio.py --debug): saves a canvas
                    # image so rendering can be inspected without a visible window.
                    data = self._read_json(limit=40_000_000)
                    name = str(data.get("name", ""))
                    if not re.fullmatch(r"[a-z0-9_-]{1,40}", name):
                        raise engine.StudioError("snapshot name must be [a-z0-9_-]{1,40}")
                    url = str(data.get("data_url", ""))
                    prefix = "data:image/png;base64,"
                    if not url.startswith(prefix):
                        raise engine.StudioError("snapshot must be a PNG data URL")
                    QA_ROOT.mkdir(parents=True, exist_ok=True)
                    target = QA_ROOT / f"{name}.png"
                    target.write_bytes(base64.b64decode(url[len(prefix):]))
                    self._send_json({"ok": True, "path": str(target)})
                elif path.startswith("/api/jobs/") and path.rsplit("/", 1)[-1] in (
                        "cancel", "pause", "resume"):
                    action = path.rsplit("/", 1)[-1]
                    job = studio.jobs.get(path.split("/")[3])
                    if job is None:
                        self._error("no such job", HTTPStatus.NOT_FOUND)
                    else:
                        getattr(job.cancel, {"cancel": "set"}.get(action, action))()
                        self._send_json({"ok": True, "job": job.snapshot()})
                else:
                    self._error("not found", HTTPStatus.NOT_FOUND)
            except engine.StudioError as exc:
                self._error(str(exc))
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self._error(f"internal error: {exc}", HTTPStatus.INTERNAL_SERVER_ERROR)

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8765, debug: bool = False) -> ThreadingHTTPServer:
    studio = Studio(debug=debug)
    server = ThreadingHTTPServer((host, port), make_handler(studio))
    server.daemon_threads = True
    server.studio = studio
    return server
