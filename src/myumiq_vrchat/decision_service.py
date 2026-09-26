"""One local model process, bounded requests, no external bind or image URLs."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .decision import BackendConfig, DecisionInput, make_scorer


def serve(config, port):
    scorer = make_scorer(config)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                {"backend": config.backend, "state": "ready"},
            )

        def do_POST(self):
            if self.path != "/score":
                self.reply(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 13_000_000:
                    raise ValueError("request size")
                self.connection.settimeout(5)
                request = DecisionInput.model_validate_json(self.rfile.read(length))
                result = scorer.score(request)
                self.reply(200, result.model_dump(mode="json"))
            except Exception as exc:
                # Avoid echoing private state, base64 images, credentials or filesystem paths.
                self.reply(422, {"error": type(exc).__name__})

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(json.dumps({"state": "ready", "port": port, "backend": config.backend}), flush=True)
        try:
            server.serve_forever()
        finally:
            if hasattr(scorer, "close"):
                scorer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18520)
    args = parser.parse_args()
    serve(BackendConfig.model_validate_json(args.config.read_text("utf-8-sig")), args.port)


if __name__ == "__main__":
    main()
