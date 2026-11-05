#!/usr/bin/env python3
"""
ThermalForge Studio -- launch the app.

    python run_studio.py                 # starts on http://127.0.0.1:8765 and opens the browser
    python run_studio.py --port 9000
    python run_studio.py --no-browser

Runs fully offline: standard-library server, no external downloads.
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
import webbrowser


def _is_in_use(host: str, port: int) -> bool:
    """True when something already answers on this address."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.4)
        return probe.connect_ex((host, port)) == 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ThermalForge Studio: thermal co-optimization "
                                                 "for 3D-IC chiplet packages.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--debug", action="store_true",
                        help="developer mode: enables the canvas snapshot endpoint")
    args = parser.parse_args(argv)

    # Double-clicking the app twice should not start a second copy. Windows
    # lets a second process bind a port another one already holds (both
    # servers set SO_REUSEADDR), and requests then land on whichever socket
    # wins -- so the port is probed first rather than trusted to fail.
    url = f"http://{args.host}:{args.port}/"
    if _is_in_use(args.host, args.port):
        print(f"ThermalForge Studio is already running at {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return 0

    # Imported here, not at module level: in a packaged build this pulls in
    # NumPy and SciPy, which takes a few seconds, and a trade-show laptop
    # should show a line of text before it goes quiet.
    print("ThermalForge Studio -- starting the solver...", flush=True)
    from studio.server import serve

    try:
        server = serve(args.host, args.port, debug=args.debug)
    except OSError as exc:
        print(f"Could not start on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"ThermalForge Studio is running at {url}")
    print("Leave this window open -- closing it stops the app.")
    print("Press Ctrl+C to stop.")
    print()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
