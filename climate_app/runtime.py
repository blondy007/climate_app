"""Supervise the web server and a session-independent forecast collector."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time


def main():
    web = subprocess.Popen([sys.executable, "-m", "climate_app.web", "run", "app.py",
        "--server.address=0.0.0.0", "--server.port=" + os.environ.get("PORT", "8501"),
        "--server.headless=true", "--browser.gatherUsageStats=false"])
    worker = subprocess.Popen([sys.executable, "-m", "climate_app.forecast.worker"])
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping and web.poll() is None:
            if worker.poll() is not None:
                time.sleep(3)
                worker = subprocess.Popen([sys.executable, "-m", "climate_app.forecast.worker"])
            time.sleep(1)
    finally:
        processes = [web, worker]
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    return web.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
