"""Cassettes: recorded LLM interactions, stored as JSONL.

A *request* here is the small dict every harness ends up building:

    {"model": "gpt-5-mini",
     "messages": [{"role": "user", "content": "What is 17 + 25?"}],
     "temperature": 0}

Its *fingerprint* is a SHA-256 over a canonical form: model, message
roles and stripped contents, and temperature. Incidental fields
(`user`, `metadata`, request ids, ...) never reach the fingerprint,
so the same logical call matches across runs even when the harness
attaches different bookkeeping to it.

A cassette file holds one interaction per line:

    {"fingerprint": "…", "request": {…canonical…},
     "response": {"text": "42", "prompt_tokens": 12,
                  "completion_tokens": 1}}

Deliberately absent: timestamps and raw headers. A cassette is a test
fixture — it should diff cleanly in code review, so the same recording
session produces the same file bytes every time.

Playback consumes interactions **in recorded order**: when the same
request was recorded three times with three different responses (an
agent loop that re-asks as its context grows), playback returns them
in that order, and a fourth identical request is a miss. This matches
how VCR-style tools behave and keeps replays honest: a test that
drifts from the recording fails loudly instead of silently reusing a
stale answer forever.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

PathLike = Union[str, Path]


class CassetteMiss(Exception):
    """Raised when playback has no (remaining) interaction for a request."""


def canonical_request(request: dict) -> dict:
    """The fingerprint-relevant part of a request, normalized."""
    messages = []
    for message in request.get("messages", []):
        messages.append(
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")).strip(),
            }
        )
    return {
        "model": request.get("model"),
        "messages": messages,
        "temperature": request.get("temperature", 0),
    }


def fingerprint_request(request: dict) -> str:
    canonical = canonical_request(request)
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Interaction:
    fingerprint: str
    request: dict
    response: dict  # {"text": str, "prompt_tokens": int|None, "completion_tokens": int|None}

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "request": self.request,
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict, where: str = "interaction") -> "Interaction":
        for name in ("fingerprint", "request", "response"):
            if name not in data:
                raise ValueError(f"{where}: missing field {name!r}")
        response = data["response"]
        if not isinstance(response, dict) or "text" not in response:
            raise ValueError(f"{where}: response must be an object with 'text'")
        return cls(
            fingerprint=str(data["fingerprint"]),
            request=data["request"],
            response={
                "text": str(response["text"]),
                "prompt_tokens": (
                    int(response["prompt_tokens"])
                    if response.get("prompt_tokens") is not None
                    else None
                ),
                "completion_tokens": (
                    int(response["completion_tokens"])
                    if response.get("completion_tokens") is not None
                    else None
                ),
            },
        )


class Cassette:
    """An ordered collection of interactions with consuming playback."""

    def __init__(self, interactions: Optional[list[Interaction]] = None) -> None:
        self._interactions: list[Interaction] = list(interactions or [])
        self._consumed: dict[str, int] = defaultdict(int)

    def __len__(self) -> int:
        return len(self._interactions)

    @property
    def interactions(self) -> list[Interaction]:
        return list(self._interactions)

    @classmethod
    def load(cls, path: PathLike) -> "Cassette":
        interactions: list[Interaction] = []
        with Path(path).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: invalid JSON: {exc}"
                    ) from exc
                interaction = Interaction.from_dict(
                    data, where=f"{path}:{lineno}"
                )
                expected = fingerprint_request(interaction.request)
                if interaction.fingerprint != expected:
                    raise ValueError(
                        f"{path}:{lineno}: fingerprint mismatch (stored "
                        f"{interaction.fingerprint[:12]}…, recomputed "
                        f"{expected[:12]}…) — the cassette was edited or "
                        f"corrupted"
                    )
                interactions.append(interaction)
        return cls(interactions)

    def save(self, path: PathLike) -> None:
        with Path(path).open("w", encoding="utf-8") as fh:
            for interaction in self._interactions:
                fh.write(
                    json.dumps(interaction.to_dict(), ensure_ascii=False) + "\n"
                )

    def append(self, request: dict, response: dict) -> Interaction:
        interaction = Interaction(
            fingerprint=fingerprint_request(request),
            request=canonical_request(request),
            response={
                "text": str(response["text"]),
                "prompt_tokens": response.get("prompt_tokens"),
                "completion_tokens": response.get("completion_tokens"),
            },
        )
        self._interactions.append(interaction)
        return interaction

    def consume(self, request: dict) -> dict:
        """Return the next unplayed response for this request, in
        recorded order. Raises CassetteMiss when none remains."""
        fingerprint = fingerprint_request(request)
        index = self._consumed[fingerprint]
        matches = [
            i for i in self._interactions if i.fingerprint == fingerprint
        ]
        if index >= len(matches):
            raise CassetteMiss(
                f"no recorded interaction left for fingerprint "
                f"{fingerprint[:12]}… (model "
                f"{canonical_request(request).get('model')!r}); "
                f"{len(matches)} recorded, all already played"
            )
        self._consumed[fingerprint] = index + 1
        return dict(matches[index].response)

    def stats(self) -> dict:
        models: dict[str, int] = defaultdict(int)
        prompt_tokens = 0
        completion_tokens = 0
        for interaction in self._interactions:
            models[str(interaction.request.get("model"))] += 1
            prompt_tokens += interaction.response.get("prompt_tokens") or 0
            completion_tokens += (
                interaction.response.get("completion_tokens") or 0
            )
        return {
            "interactions": len(self._interactions),
            "unique_fingerprints": len(
                {i.fingerprint for i in self._interactions}
            ),
            "models": dict(models),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
