# Changelog

## 0.1.0 — 2026-10-31

- **Initial release: record/replay for LLM calls.** A cassette is a
  JSONL file of interactions keyed by a SHA-256 fingerprint of the
  canonical request (model, message roles and stripped contents,
  temperature; incidental fields excluded), verified on load and
  free of timestamps so cassettes diff as fixtures. Playback consumes
  interactions in recorded order per fingerprint and raises
  `CassetteMiss` when exhausted. `CassetteClient` wraps any live-call
  callable in replay / record / auto modes with hit, miss and
  live-call counters. The `serve` subcommand runs an
  OpenAI-compatible endpoint from a cassette (misses return a named
  `cassette_miss` 404; `/healthz` reports the load), and with
  `--upstream` becomes a recording proxy that forwards requests,
  relays responses verbatim and appends them to the cassette.
  `stats` validates and summarizes a cassette; `demo` records a tiny
  scripted session and replays it with zero live calls. Standard
  library only.
