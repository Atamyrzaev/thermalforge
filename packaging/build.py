"""
packaging/build.py
RISC3D-AI :: build the standalone app.

Produces dist/RISC3D Studio/, a folder containing everything a machine needs
to run the app -- Python, NumPy, SciPy, the solver, the UI and the measured
validation artifacts. Copy the folder to a laptop that has never seen Python,
double-click "RISC3D Studio.exe", and the browser app opens.

    python packaging/build.py                # build
    python packaging/build.py --check        # build, then start it and solve
                                             # a scenario through the real exe

Requires PyInstaller (pip install pyinstaller). Everything else the build
needs is already a dependency of the app.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "risc3d_studio.spec"
BUNDLE = ROOT / "dist" / "RISC3D Studio"
EXE = BUNDLE / "RISC3D Studio.exe"
CHECK_PORT = 8799


def build() -> None:
    if not SPEC.exists():
        raise SystemExit(f"missing spec: {SPEC}")
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit("PyInstaller is not installed: pip install pyinstaller")
    print("Building... (a couple of minutes the first time)")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-m", "PyInstaller", str(SPEC), "--noconfirm",
                    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build")],
                   cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    size_mb = sum(f.stat().st_size for f in BUNDLE.rglob("*") if f.is_file()) / 1e6
    print(f"Built {BUNDLE} -- {size_mb:.0f} MB in {time.perf_counter() - t0:.0f} s")


def check() -> None:
    """Start the built exe and make it solve, through HTTP, like a user would.

    A build that imports cleanly can still be missing a data file the app only
    reads at runtime, so this asks the running exe for a real answer and
    checks the verification that comes back with it.
    """
    if not EXE.exists():
        raise SystemExit(f"no build to check: {EXE}")
    print(f"Starting {EXE.name}...")
    proc = subprocess.Popen([str(EXE), "--no-browser", "--port", str(CHECK_PORT)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{CHECK_PORT}"
    t0 = time.perf_counter()
    try:
        scenarios = None
        for _ in range(120):
            if proc.poll() is not None:
                raise SystemExit(f"the app exited immediately:\n{proc.stdout.read()[:2000]}")
            try:
                scenarios = json.loads(urllib.request.urlopen(base + "/api/scenarios", timeout=3).read())
                break
            except Exception:
                time.sleep(0.5)
        if not scenarios or not scenarios.get("ok"):
            raise SystemExit("the app started but never served its scenarios")
        print(f"  answered in {time.perf_counter() - t0:.1f} s, "
              f"{len(scenarios['scenarios'])} scenarios")

        for path in ("/", "/css/app.css", "/js/app.js", "/js/design.js", "/js/report.js",
                     "/js/viewer3d.js", "/js/mapdraw.js"):
            status = urllib.request.urlopen(base + path, timeout=10).status
            if status != 200:
                raise SystemExit(f"  {path} returned {status}")
        print("  every asset the page needs is served")

        first = scenarios["scenarios"][0]
        chiplets = [{k: v for k, v in c.items() if k != "color"} for c in first["chiplets"]]
        request = urllib.request.Request(
            base + "/api/solve",
            data=json.dumps({"package": first["package"], "chiplets": chiplets}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        solved = json.loads(urllib.request.urlopen(request, timeout=180).read())
        if not solved.get("ok"):
            raise SystemExit(f"  the solve failed: {solved.get('error') or solved.get('violations')}")
        v = solved["verification"]
        if v["residual"] >= 1e-8:
            raise SystemExit(f"  residual {v['residual']:.1e} is not a solved system")
        if abs(v["sink_heat_w"] - v["specified_w"]) > 1e-6 * max(v["specified_w"], 1e-12):
            raise SystemExit("  power in does not equal power out")
        band = solved["metrics"].get("uncertainty")
        print(f"  solved {first['id']}: peak {solved['peak']['t_c']:.2f} degC"
              + (f" +/- {band['combined_k']:.2f} K" if band else " (no error budget bundled)")
              + f", residual {v['residual']:.1e}")
        print("The build works.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="after building, run the exe and solve through it")
    parser.add_argument("--clean", action="store_true", help="remove build/ and dist/ first")
    args = parser.parse_args()
    if args.clean:
        for directory in (ROOT / "build", ROOT / "dist"):
            shutil.rmtree(directory, ignore_errors=True)
    build()
    if args.check:
        check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
