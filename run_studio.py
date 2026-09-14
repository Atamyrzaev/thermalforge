#!/usr/bin/env python3
"""
RISC3D Studio -- launch the app.

    python run_studio.py                 # starts on http://127.0.0.1:8765 and opens the browser
    python run_studio.py --port 9000
    python run_studio.py --no-browser

Runs fully offline: standard-library server, no external downloads.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

from studio.server import serve


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RISC3D Studio: thermal co-optimization "
                                                 "for 3D-IC chiplet packages.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true",
                        help="do not open a browser window")
    parser.add_argument("--debug", action="store_true",
                        help="developer mode: enables the canvas snapshot endpoint")
    args = parser.parse_args(argv)

    try:
        server = serve(args.host, args.port, debug=args.debug)
    except OSError as exc:
        print(f"Could not start on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"RISC3D Studio is running at {url}")
    print("Press Ctrl+C to stop.")
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
