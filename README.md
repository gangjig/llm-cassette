# llm-cassette

**VCR for LLM calls: record an agent's model calls once, replay them
in tests forever — offline, deterministic, free.**

Every agent harness eventually needs tests, and every harness test
that calls a live model is slow, flaky, and metered. The usual escape
is hand-rolled mocks, which drift from reality the day they're
written. `llm-cassette` takes the VCR approach from HTTP testing:
record real interactions into a JSONL **cassette file**, commit it
with your tests, and replay it in CI. If a prompt changes and the
cassette misses, the test fails loudly at the call — which is exactly
where you want to know.

## Quickstart

No keys, no network — the demo records a tiny session against a
scripted fake "live API", then replays it:

```bash
pip install -e .
llm-cassette demo
```

```
Run 1 (auto mode, empty cassette):
  Q: What is 17 + 25?  ->  42
  ...
  live calls: 3, cassette saved with 3 interaction(s)
Run 2 (replay mode, cassette loaded):
  Q: What is 17 + 25?  ->  42
  ...
  live calls: 0
Answers identical across runs: yes
```

Use it as a library around whatever call shape your harness already
has:

```python
from llm_cassette import Cassette, CassetteClient

client = CassetteClient(my_live_call, Cassette.load("tests/cassette.jsonl"),
                        mode="replay")   # never touches the network
response = client.complete({"model": "gpt-5-mini",
                            "messages": [{"role": "user", "content": "..."}],
                            "temperature": 0})
```

Or skip code entirely and point your harness's `base_url` at the
built-in OpenAI-compatible server:

```bash
llm-cassette serve --cassette tests/cassette.jsonl --port 8787
# record a fresh cassette against a real API instead:
llm-cassette serve --cassette tests/cassette.jsonl --port 8787 \
    --upstream https://api.openai.com/v1
```

## How it works

**Fingerprints.** A request is identified by a SHA-256 over its
canonical form: model, message roles and stripped contents, and
temperature. Incidental fields — `user`, `metadata`, request ids —
never reach the fingerprint, so the same logical call matches across
runs even when your harness attaches different bookkeeping.

**Ordered playback.** A cassette can hold the same request several
times with different responses (an agent loop re-asking as its
context grows). Playback returns them in recorded order, and one
request too many is a `CassetteMiss` — a test that drifts from its
recording fails instead of silently reusing a stale answer forever.

**Modes.** `replay` never calls live (the CI mode). `record` always
calls live and appends (the fixture-creation mode). `auto` replays on
a hit and records on a miss (comfortable while developing a test,
too forgiving for CI). The client counts hits, misses and live calls,
so a test can *prove* a replay made zero live calls.

**Cassettes are fixtures, not logs.** One interaction per JSONL line,
fingerprints verified on load, and deliberately no timestamps or
headers — the same recording session produces the same file bytes, so
cassettes diff cleanly in code review. `llm-cassette stats --cassette
FILE` validates a cassette and summarizes its interactions, models
and recorded tokens.

**The server** speaks just enough of the OpenAI chat-completions
shape for a harness to talk to: `POST …/chat/completions` answers
from the cassette (a miss returns 404 with a `cassette_miss` error
body), `GET /healthz` reports the loaded interaction count. With
`--upstream` it becomes a recording proxy: requests are forwarded
(your Authorization header goes upstream and is never stored), the
response is relayed verbatim, and parseable responses are appended
to the cassette file.

## Recreate this project from a prompt

> Build a Python package called `llm-cassette`: VCR-style record and
> replay for LLM chat-completion calls, standard library only. A
> request is a dict with model, messages and temperature; its
> fingerprint is a SHA-256 over a canonical JSON form (model, message
> roles and stripped contents, temperature — incidental fields like
> user/metadata excluded). A cassette is a JSONL file of interactions
> (fingerprint, canonical request, response with text and optional
> token counts), verified against recomputed fingerprints on load,
> with no timestamps so files diff cleanly. Playback consumes
> interactions in recorded order per fingerprint and raises a
> CassetteMiss when exhausted. Provide a client wrapper around any
> live-call callable with three modes — replay (never live, miss
> raises), record (always live, append), auto (replay on hit, record
> on miss) — and hit/miss/live-call counters. Provide an
> OpenAI-compatible HTTP server (stdlib http.server) that answers
> chat-completion POSTs from a cassette with a proper response body,
> returns a named `cassette_miss` 404 on a miss, exposes /healthz,
> and with an --upstream option acts as a recording proxy that
> forwards requests, relays responses verbatim and appends parseable
> responses to the cassette file. Add a CLI with serve, stats and a
> demo subcommand (record a tiny scripted session against a fake live
> call, then replay it with zero live calls and identical answers),
> and a pytest suite covering fingerprint canonicalization, ordered
> playback, client modes, the HTTP server in both modes against real
> local sockets, and the CLI.

## Limitations

- **Chat completions, text only.** The fingerprint and response shape
  cover model/messages/temperature and a text answer with token
  counts. Streaming, tool calls and multimodal content are not
  recorded yet — a request containing them will fingerprint only its
  text parts, which may collide; check `stats` before trusting a
  cassette recorded from such traffic.
- **Exact-match replay.** There is no fuzzy matching: a one-character
  prompt change is a miss. That strictness is the point in tests, but
  it means prompt templates with embedded timestamps or ids must be
  stabilized by the harness before the cassette can help.
- **The recording proxy stores what the API returned** — review a
  cassette before committing it, the way you would review any fixture
  that may contain real model outputs.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e . pytest
python -m pytest        # 23 tests, fully offline
```

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Gang Ji.
