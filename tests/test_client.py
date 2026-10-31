"""The client wrapper: replay/record/auto semantics and counters."""

import pytest

from llm_cassette.cassette import Cassette, CassetteMiss
from llm_cassette.client import CassetteClient

REQUEST = {
    "model": "m",
    "messages": [{"role": "user", "content": "What is 17 + 25?"}],
    "temperature": 0,
}


def make_live():
    calls = []

    def live(request):
        calls.append(request)
        return {"text": "42", "prompt_tokens": 10, "completion_tokens": 1}

    return live, calls


def test_replay_hit_never_calls_live():
    live, calls = make_live()
    cassette = Cassette()
    cassette.append(REQUEST, {"text": "42", "prompt_tokens": 10, "completion_tokens": 1})
    client = CassetteClient(live, cassette, mode="replay")
    assert client.complete(REQUEST)["text"] == "42"
    assert calls == []
    assert client.hits == 1 and client.live_calls == 0


def test_replay_miss_raises_and_never_calls_live():
    live, calls = make_live()
    client = CassetteClient(live, Cassette(), mode="replay")
    with pytest.raises(CassetteMiss):
        client.complete(REQUEST)
    assert calls == []
    assert client.misses == 1


def test_auto_records_then_replays():
    live, calls = make_live()
    client = CassetteClient(live, Cassette(), mode="auto")
    assert client.complete(REQUEST)["text"] == "42"
    assert client.complete(REQUEST)["text"] == "42"
    assert len(calls) == 1  # second call was a cassette hit
    assert client.live_calls == 1 and client.hits == 1


def test_record_mode_always_calls_live_and_appends():
    live, calls = make_live()
    cassette = Cassette()
    cassette.append(REQUEST, {"text": "stale", "prompt_tokens": 1, "completion_tokens": 1})
    client = CassetteClient(live, cassette, mode="record")
    assert client.complete(REQUEST)["text"] == "42"
    assert len(calls) == 1
    assert len(cassette) == 2


def test_invalid_mode_rejected():
    live, _ = make_live()
    with pytest.raises(ValueError, match="mode must be one of"):
        CassetteClient(live, Cassette(), mode="sometimes")
