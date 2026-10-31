"""Fingerprints and the cassette store: canonicalization, ordered
playback, integrity checking."""

import pytest

from llm_cassette.cassette import (
    Cassette,
    CassetteMiss,
    canonical_request,
    fingerprint_request,
)


def req(content="What is 17 + 25?", model="m", temperature=0, **extra):
    base = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": temperature,
    }
    base.update(extra)
    return base


def resp(text="42"):
    return {"text": text, "prompt_tokens": 10, "completion_tokens": 1}


def test_fingerprint_ignores_incidental_fields_and_key_order():
    a = req(user="alice", metadata={"run": 1})
    b = {
        "temperature": 0,
        "messages": [{"content": "What is 17 + 25?", "role": "user"}],
        "model": "m",
        "request_id": "xyz",
    }
    assert fingerprint_request(a) == fingerprint_request(b)


def test_fingerprint_normalizes_content_whitespace():
    assert fingerprint_request(req("  hello ")) == fingerprint_request(req("hello"))


def test_fingerprint_changes_with_model_temperature_content():
    base = fingerprint_request(req())
    assert fingerprint_request(req(model="other")) != base
    assert fingerprint_request(req(temperature=1)) != base
    assert fingerprint_request(req("different question")) != base


def test_canonical_request_defaults_temperature_zero():
    no_temp = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    assert canonical_request(no_temp)["temperature"] == 0
    assert fingerprint_request(no_temp) == fingerprint_request(req("x"))


def test_save_load_roundtrip(tmp_path):
    cassette = Cassette()
    cassette.append(req(), resp())
    cassette.append(req("capital?"), resp("Paris"))
    path = tmp_path / "c.jsonl"
    cassette.save(path)
    loaded = Cassette.load(path)
    assert len(loaded) == 2
    assert loaded.consume(req())["text"] == "42"


def test_consume_plays_back_in_recorded_order():
    cassette = Cassette()
    request = req()
    cassette.append(request, resp("first"))
    cassette.append(request, resp("second"))
    assert cassette.consume(request)["text"] == "first"
    assert cassette.consume(request)["text"] == "second"
    with pytest.raises(CassetteMiss, match="already played"):
        cassette.consume(request)


def test_consume_unknown_request_misses():
    cassette = Cassette()
    with pytest.raises(CassetteMiss, match="no recorded interaction"):
        cassette.consume(req())


def test_load_rejects_corrupt_fingerprint(tmp_path):
    cassette = Cassette()
    cassette.append(req(), resp())
    path = tmp_path / "c.jsonl"
    cassette.save(path)
    text = path.read_text(encoding="utf-8").replace("42", "43")
    path.write_text(text, encoding="utf-8")
    # The stored response changed, but fingerprints cover the request,
    # so instead corrupt the request content:
    loaded_text = path.read_text(encoding="utf-8")
    assert "43" in loaded_text
    path.write_text(
        loaded_text.replace("What is 17 + 25?", "What is 18 + 25?"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        Cassette.load(path)


def test_load_rejects_invalid_json_and_missing_fields(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        Cassette.load(path)
    path.write_text('{"fingerprint": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing field"):
        Cassette.load(path)


def test_stats():
    cassette = Cassette()
    cassette.append(req(), resp())
    cassette.append(req(), resp())
    cassette.append(req("capital?", model="other"), resp("Paris"))
    stats = cassette.stats()
    assert stats["interactions"] == 3
    assert stats["unique_fingerprints"] == 2
    assert stats["models"] == {"m": 2, "other": 1}
    assert stats["prompt_tokens"] == 30
