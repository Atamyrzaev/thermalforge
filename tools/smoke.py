"""
tools/smoke.py
RISC3D-AI :: does a released build actually start and answer?

Starts the Studio server on an ephemeral port, asks it for the scenarios,
solves one and checks the answer carries its verification with it. This is
the check that fails when the app is broken in a way unit tests cannot see:
a missing static file, an unimportable module, a server that starts and then
refuses every request.

    python tools/smoke.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from studio.server import serve  # noqa: E402

TIMEOUT_S = 60


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


def main() -> int:
    server = serve("127.0.0.1", 0)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    failures = []
    try:
        # The page and every asset the page needs.
        for path in ("/", "/css/app.css", "/js/app.js", "/js/design.js", "/js/report.js"):
            with urllib.request.urlopen(base + path, timeout=TIMEOUT_S) as response:
                if response.status != 200 or not response.read():
                    failures.append(f"{path} did not serve")

        scenarios = get(base + "/api/scenarios")
        if not scenarios.get("ok") or not scenarios.get("scenarios"):
            failures.append("no scenarios")
            raise SystemExit(1)

        first = scenarios["scenarios"][0]
        chiplets = [{k: v for k, v in c.items() if k != "color"} for c in first["chiplets"]]
        t0 = time.perf_counter()
        solved = post(base + "/api/solve", {"package": first["package"], "chiplets": chiplets})
        seconds = time.perf_counter() - t0
        if not solved.get("ok"):
            failures.append(f"solve failed: {solved.get('error') or solved.get('violations')}")
        else:
            v = solved["verification"]
            if v["residual"] >= 1e-8:
                failures.append(f"residual {v['residual']:.1e} is not a solved system")
            if abs(v["sink_heat_w"] - v["specified_w"]) > 1e-6 * max(v["specified_w"], 1e-12):
                failures.append("power in does not equal power out")
            print(f"{first['id']}: peak {solved['peak']['t_c']:.2f} degC, "
                  f"residual {v['residual']:.1e}, {seconds:.1f} s including model build")
    finally:
        server.shutdown()

    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print("smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
