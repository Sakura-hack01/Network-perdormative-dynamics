"""Interactive Web Dashboard server for Networked Performative Dynamics.

Run with:
    python app.py
Then navigate to http://localhost:8000 in your browser.
"""

import http.server
import socketserver
import os
import sys
import json
import urllib.parse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


class ResearchDashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving dashboard GUI, figures, and simulation API."""

    def translate_path(self, path):
        # Normalize path
        clean_path = path.split("?", 1)[0].split("#", 1)[0]
        if clean_path in ["", "/", "/index.html"]:
            return str(REPO_ROOT / "interface" / "index.html")
        elif clean_path.startswith("/figures/"):
            rel = clean_path.replace("/figures/", "")
            return str(REPO_ROOT / "figures" / rel)
        elif clean_path.startswith("/results/"):
            rel = clean_path.replace("/results/", "")
            return str(REPO_ROOT / "results" / rel)
        elif clean_path.startswith("/interface/"):
            rel = clean_path.replace("/interface/", "")
            return str(REPO_ROOT / "interface" / rel)
        return super().translate_path(path)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            status = {
                "status": "online",
                "framework": "PyTorch / NumPy Performative Dynamics",
                "experiments_available": ["exp1", "exp2", "exp3", "exp4"],
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return
        super().do_GET()


def run_server(port: int = 8000, host: str = "127.0.0.1"):
    """Starts the HTTP server."""
    os.chdir(str(REPO_ROOT))
    handler = ResearchDashboardHandler
    with socketserver.TCPServer((host, port), handler) as httpd:
        print(f"\n=======================================================")
        print(f"  Networked Performative Dynamics - Research Dashboard")
        print(f"  Serving at: http://{host}:{port}/")
        print(f"  Press Ctrl+C to terminate the server.")
        print(f"=======================================================\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port=port)
