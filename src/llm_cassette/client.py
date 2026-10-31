"""Wrap a live LLM call so tests can replay it instead of paying for it.

    client = CassetteClient(live_call, Cassette.load("tests/cassette.jsonl"),
                            mode="replay")
    response = client.complete({"model": "gpt-5-mini", "messages": [...]})

`live_call` is any callable taking the request dict and returning
{"text": ..., "prompt_tokens": ..., "completion_tokens": ...} — the
shape every provider SDK can be adapted to in three lines.

Modes:

* `replay` -- never call live; a miss raises CassetteMiss. The mode
  CI should use: tests are offline, deterministic and free, and a
  prompt change that misses the cassette fails loudly.
* `record` -- always call live and append the interaction. Use when
  (re)creating a fixture, then commit the cassette file.
* `auto`   -- replay on a hit, call live and append on a miss. Handy
  while developing a test; too forgiving for CI, where a silent live
  call is exactly what you are trying to prevent.

The client counts hits, misses and live calls, so a test (or the
demo) can prove that a replay really made zero live calls.
"""

from __future__ import annotations

from typing import Callable

from .cassette import Cassette, CassetteMiss

MODES = ("replay", "record", "auto")

LiveCall = Callable[[dict], dict]


class CassetteClient:
    def __init__(
        self, live_call: LiveCall, cassette: Cassette, mode: str = "replay"
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._live_call = live_call
        self.cassette = cassette
        self.mode = mode
        self.hits = 0
        self.misses = 0
        self.live_calls = 0

    def complete(self, request: dict) -> dict:
        if self.mode in ("replay", "auto"):
            try:
                response = self.cassette.consume(request)
                self.hits += 1
                return response
            except CassetteMiss:
                self.misses += 1
                if self.mode == "replay":
                    raise
        response = self._live_call(request)
        self.live_calls += 1
        self.cassette.append(request, response)
        return dict(response)
