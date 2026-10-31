"""CLI: demo proves the point; stats validates and summarizes."""

import json

from llm_cassette.cassette import Cassette
from llm_cassette.cli import main
from llm_cassette.demo import REQUESTS


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "live calls: 3" in out  # run 1 recorded everything live
    assert "live calls: 0" in out  # run 2 replayed everything
    assert "Answers identical across runs: yes" in out


def test_cli_stats_markdown_and_json(tmp_path, capsys):
    cassette = Cassette()
    for request in REQUESTS:
        cassette.append(
            request, {"text": "x", "prompt_tokens": 5, "completion_tokens": 2}
        )
    path = tmp_path / "c.jsonl"
    cassette.save(path)
    assert main(["stats", "--cassette", str(path)]) == 0
    out = capsys.readouterr().out
    assert "interactions: 3" in out
    assert "demo-model × 3" in out
    assert main(["stats", "--cassette", str(path), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["interactions"] == 3
    assert data["prompt_tokens"] == 15


def test_cli_stats_missing_file_returns_2(tmp_path, capsys):
    assert main(["stats", "--cassette", str(tmp_path / "nope.jsonl")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_serve_replay_missing_cassette_returns_2(tmp_path, capsys):
    assert main(["serve", "--cassette", str(tmp_path / "nope.jsonl")]) == 2
    assert "not found" in capsys.readouterr().err
