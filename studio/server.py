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

from studio import engine
from studio.scenarios import scenarios

STATIC_ROOT = Path(__file__).resolve().parent / "static"
QA_ROOT = Path(__file__).resolve().parent.parent / "outputs" / "studio_qa"
MAX_BODY_BYTES = 1_000_000
MAX_JOBS = 16

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("image/svg+xml", ".svg")


class Job:
    def __init__(self):
        self.id = uuid.uuid4().hex[:12]
        self.created = time.time()
        self.state = "running"
        self.progress: Dict = {"stage": "queued", "elapsed_s": 0.0}
        self.result: Optional[dict] = None
        self.error: Optional[str] = None
        self.cancel = threading.Event()
        self.lock = threading.Lock()

    def snapshot(self) -> dict:
        with self.lock:
            out = {"id": self.id, "state": self.state, "progress": dict(self.progress)}
            if self.state == "done":
                out["result"] = self.result
            if self.error:
                out["error"] = self.error
            return out


class Studio:
    """Shared state: the model cache and the optimization jobs."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.cache = engine.ModelCache()
        self.jobs: Dict[str, Job] = {}
        self.jobs_lock = threading.Lock()

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
                elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                    job = studio.jobs.get(path.split("/")[3])
                    if job is None:
                        self._error("no such job", HTTPStatus.NOT_FOUND)
                    else:
                        job.cancel.set()
                        self._send_json({"ok": True})
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
