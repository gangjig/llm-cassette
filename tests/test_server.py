"""The HTTP server: replay over the wire, and the recording proxy
against a real local upstream."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_cassette.cassette import Cassette
from llm_cassette.server import make_server

REQUEST = {
    "model": "m",
    "messages": [{"role": "user", "content": "What is 17 + 25?"}],
    "temperature": 0,
}


def _post(port, payload, path="/v1/chat/completions"):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture()
def replay_server():
    cassette = Cassette()
    cassette.append(
        REQUEST, {"text": "42", "prompt_tokens": 10, "completion_tokens": 1}
    )
    server = make_server(cassette)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_replay_server_answers_recorded_request(replay_server):
    status, body = _post(replay_server, REQUEST)
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "42"
    assert body["usage"]["total_tokens"] == 11
    assert body["model"] == "m"


def test_replay_server_miss_is_a_named_404(replay_server):
    status, body = _post(replay_server, {**REQUEST, "model": "other"})
    assert status == 404
    assert body["error"]["type"] == "cassette_miss"


def test_replay_server_healthz(replay_server):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{replay_server}/healthz", timeout=5
    ) as response:
        body = json.loads(response.read())
    assert body["status"] == "ok"
    assert body["interactions"] == 1
    assert body["mode"] == "replay"


class _UpstreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = {
            "choices": [{"message": {"content": "42"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture()
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server.server_address[1]
    server.shutdown()


def test_record_proxy_forwards_and_records(tmp_path, upstream):
    cassette_path = tmp_path / "recorded.jsonl"
    server = make_server(
        Cassette(),
        cassette_path=str(cassette_path),
        upstream=f"http://127.0.0.1:{upstream}/v1",
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status, body = _post(server.server_address[1], REQUEST)
        assert status == 200
        assert body["choices"][0]["message"]["content"] == "42"
    finally:
        server.shutdown()
    recorded = Cassette.load(cassette_path)
    assert len(recorded) == 1
    assert recorded.consume(REQUEST)["text"] == "42"
