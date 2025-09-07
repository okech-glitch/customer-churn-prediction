import os
import subprocess
import sys
import time
import urllib.request

PORT = 8000


def test_backend_health_endpoint_starts_and_reports_healthy():
    # Start the API in a subprocess
    proc = subprocess.Popen([sys.executable, os.path.join('backend', 'main.py')])
    try:
        # Wait briefly for startup
        time.sleep(2.5)
        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5) as resp:
            assert resp.status == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
