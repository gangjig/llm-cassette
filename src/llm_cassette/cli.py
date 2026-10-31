"""Command line interface: llm-cassette {serve, stats, demo}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .cassette import Cassette
from .demo import run_demo
from .server import make_server


def _print_or_write(text: str, output: Optional[str]) -> None:
    if output:
        Path(output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-cassette",
        description="Record and replay LLM API calls for deterministic, "
        "offline agent-harness tests.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser(
        "serve",
        help="serve chat completions from a cassette (replay), or as a "
        "recording proxy with --upstream",
    )
    p_serve.add_argument("--cassette", required=True, metavar="FILE")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8787)
    p_serve.add_argument(
        "--upstream",
        metavar="URL",
        help="record mode: forward requests to this base URL and append "
        "the responses to the cassette",
    )

    p_stats = sub.add_parser(
        "stats", help="validate a cassette and summarize its contents"
    )
    p_stats.add_argument("--cassette", required=True, metavar="FILE")
    p_stats.add_argument(
        "--format", choices=("md", "json"), default="md",
        help="output format (default: md)",
    )
    p_stats.add_argument(
        "--output", metavar="FILE", help="write to FILE instead of stdout"
    )

    sub.add_parser(
        "demo",
        help="record a tiny scripted session, then replay it with zero "
        "live calls (no network, no keys)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            path = Path(args.cassette)
            if args.upstream:
                cassette = Cassette.load(path) if path.exists() else Cassette()
            else:
                if not path.exists():
                    print(
                        f"error: cassette file not found: {path} "
                        f"(replay mode needs an existing cassette; "
                        f"use --upstream to record one)",
                        file=sys.stderr,
                    )
                    return 2
                cassette = Cassette.load(path)
            server = make_server(
                cassette,
                host=args.host,
                port=args.port,
                cassette_path=str(path),
                upstream=args.upstream,
            )
            mode = "record" if args.upstream else "replay"
            print(
                f"llm-cassette serving on http://{args.host}:{args.port} "
                f"({mode} mode, {len(cassette)} interaction(s) loaded)",
                file=sys.stderr,
            )
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                return 0
            return 0

        if args.command == "stats":
            cassette = Cassette.load(args.cassette)
            stats = cassette.stats()
            if args.format == "json":
                text = json.dumps(stats, indent=2) + "\n"
            else:
                lines = [
                    f"interactions: {stats['interactions']}",
                    f"unique fingerprints: {stats['unique_fingerprints']}",
                    "models: "
                    + (
                        ", ".join(
                            f"{name} × {count}"
                            for name, count in sorted(stats["models"].items())
                        )
                        or "(none)"
                    ),
                    f"recorded tokens: {stats['prompt_tokens']} prompt + "
                    f"{stats['completion_tokens']} completion",
                ]
                text = "\n".join(lines) + "\n"
            _print_or_write(text, args.output)
            return 0

        # demo
        result = run_demo()
        lines = ["Run 1 (auto mode, empty cassette):"]
        for question, answer in result["run1"]["lines"]:
            lines.append(f"  Q: {question}  ->  {answer}")
        lines.append(
            f"  live calls: {result['run1']['live_calls']}, "
            f"cassette saved with {result['interactions']} interaction(s)"
        )
        lines.append("Run 2 (replay mode, cassette loaded):")
        for question, answer in result["run2"]["lines"]:
            lines.append(f"  Q: {question}  ->  {answer}")
        lines.append(f"  live calls: {result['run2']['live_calls']}")
        lines.append(
            "Answers identical across runs: "
            + ("yes" if result["identical"] else "NO")
        )
        sys.stdout.write("\n".join(lines) + "\n")
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
