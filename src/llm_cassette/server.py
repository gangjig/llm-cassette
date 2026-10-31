"""An OpenAI-compatible HTTP endpoint backed by a cassette.

Point any harness's `base_url` at this server and its chat-completion
calls are answered from the cassette — no SDK changes, no keys, no
network:

    llm-cassette serve --cassette tests/cassette.jsonl --port 8787

Replay mode is the default: a request with no recorded interaction
left gets a 404 whose error body names the miss (type
`cassette_miss`), so a drifting test fails where it should — at the
call — instead of silently reaching a real API.

With `--upstream https://api.openai.com/v1` the same server becomes a
recording proxy: it forwards each request (including the caller's
Authorization header, which is never stored), relays the response
verbatim, and appends the interaction to the cassette file. Record
once against the real API, replay forever offline.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .cassette import Cassette, CassetteMiss, fingerprint_request


def openai_body(request: dict, response: dict) -> dict:
    prompt_tokens = response.get("prompt_tokens") or 0
    completion_tokens = response.get("completion_tokens") or 0
    return {
        "id": f"chatcmpl-cassette-{fingerprint_request(request)[:12]}",
        "object": "chat.completion",
        "created": 0,
        "model": request.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["text"],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _extract_response(payload: dict) -> dict:
    usage = payload.get("usage") or {}
    return {
        "text": str(payload["choices"][0]["message"]["content"]),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # Attached by make_server: cassette, cassette_path, upstream.
    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path.rstrip("/").endswith("healthz"):
            self._send_json(
                200,
                {
                    "status": "ok",
                    "interactions": len(self.server.cassette),
                    "mode": "record" if self.server.upstream else "replay",
                },
            )
        else:
            self._send_json(404, {"error": {"message": "not found"}})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        if not self.path.rstrip("/").endswith("chat/completions"):
            self._send_json(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return
        request = {
            "model": body.get("model"),
            "messages": body.get("messages", []),
            "temperature": body.get("temperature", 0),
        }
        if self.server.upstream:
            self._record(request, body)
        else:
            self._replay(request)

    def _replay(self, request: dict) -> None:
        try:
            response = self.server.cassette.consume(request)
        except CassetteMiss as exc:
            self._send_json(
                404,
                {
                    "error": {
                        "message": str(exc),
                        "type": "cassette_miss",
                        "fingerprint": fingerprint_request(request),
                    }
                },
            )
            return
        self._send_json(200, openai_body(request, response))

    def _record(self, request: dict, body: dict) -> None:
        upstream_url = self.server.upstream.rstrip("/") + "/chat/completions"
        forwarded = urllib.request.Request(
            upstream_url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        auth = self.headers.get("Authorization")
        if auth:
            forwarded.add_header("Authorization", auth)
        try:
            with urllib.request.urlopen(forwarded, timeout=120) as resp:
                status = resp.status
                payload_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            payload_bytes = exc.read()
        except urllib.error.URLError as exc:
            self._send_json(
                502,
                {"error": {"message": f"upstream unreachable: {exc.reason}"}},
            )
            return
        if status == 200:
            try:
                payload = json.loads(payload_bytes.decode("utf-8"))
                self.server.cassette.append(request, _extract_response(payload))
                if self.server.cassette_path:
                    self.server.cassette.save(self.server.cassette_path)
            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                pass  # relay verbatim even if it cannot be recorded
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload_bytes)))
        self.end_headers()
        self.wfile.write(payload_bytes)

    def log_message(self, *args):  # keep the console quiet
        pass


def make_server(
    cassette: Cassette,
    host: str = "127.0.0.1",
    port: int = 0,
    cassette_path: Optional[str] = None,
    upstream: Optional[str] = None,
) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), _Handler)
    server.cassette = cassette
    server.cassette_path = cassette_path
    server.upstream = upstream
    return server
