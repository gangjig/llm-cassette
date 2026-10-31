"""The demo: record a tiny agent session once, replay it with zero
live calls. The "live API" is a scripted fake, so the demo itself
needs no keys and no network.

Run 1 uses a client in `auto` mode: every request misses the empty
cassette, goes "live" (the fake), and is recorded. The cassette is
saved to a temp file. Run 2 loads it with a fresh client in `replay`
mode and issues the identical requests: every one is a cassette hit,
the live-call counter stays at zero, and the answers are byte-identical.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .cassette import Cassette
from .client import CassetteClient

SCRIPTED_ANSWERS: dict[str, str] = {
    "What is 17 + 25?": "42",
    "What is the capital of France?": "Paris",
    "Translate 'thank you' into German.": "Danke",
}

REQUESTS: list[dict] = [
    {
        "model": "demo-model",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0,
    }
    for question in SCRIPTED_ANSWERS
]


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def fake_live_call(request: dict) -> dict:
    question = request["messages"][-1]["content"]
    text = SCRIPTED_ANSWERS[question]
    return {
        "text": text,
        "prompt_tokens": _tokens(question),
        "completion_tokens": _tokens(text),
    }


def run_demo() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo-cassette.jsonl"
        recorder = CassetteClient(fake_live_call, Cassette(), mode="auto")
        run1 = []
        for request in REQUESTS:
            response = recorder.complete(request)
            run1.append((request["messages"][-1]["content"], response["text"]))
        recorder.cassette.save(path)

        player = CassetteClient(fake_live_call, Cassette.load(path), mode="replay")
        run2 = []
        for request in REQUESTS:
            response = player.complete(request)
            run2.append((request["messages"][-1]["content"], response["text"]))

        return {
            "run1": {"lines": run1, "live_calls": recorder.live_calls},
            "run2": {"lines": run2, "live_calls": player.live_calls},
            "identical": run1 == run2,
            "interactions": len(player.cassette),
        }
