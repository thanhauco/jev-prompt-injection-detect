"""Command line entry point.

    jev-pi-detect "Ignore previous instructions and ..."
    cat page.html | jev-pi-detect --source web --context "Summarize this page"
    jev-pi-detect eval data/sample.jsonl

Exit status: 0 allow, 1 review, 2 block, 3 usage/config error.
"""

from __future__ import annotations

import argparse
import json
import sys

from typesafe_sdk import TypeSafeError

from jev_pi_detect.detector import DetectorConfig, PromptInjectionDetector

_EXIT = {"allow": 0, "review": 1, "block": 2}


def _config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(model=args.model, block_threshold=args.block, review_threshold=args.review)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=None, help="Jev model (default: SDK default)")
    p.add_argument("--block", type=float, default=0.8, help="block threshold (default 0.8)")
    p.add_argument("--review", type=float, default=0.4, help="review threshold (default 0.4)")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["eval"]:
        from jev_pi_detect.evaluate import main as eval_main

        return eval_main(argv[1:])

    parser = argparse.ArgumentParser(prog="jev-pi-detect", description="Screen text for prompt injection with Jev.")
    parser.add_argument("text", nargs="?", help="text to screen (default: stdin)")
    parser.add_argument("--source", help="where the text came from (email, web, rag, tool_output, user)")
    parser.add_argument("--context", help="what the downstream LLM will do with the text")
    _add_common(parser)
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else sys.stdin.read()
    try:
        with PromptInjectionDetector(config=_config(args)) as detector:
            result = detector.detect(text, context=args.context, source=args.source)
    except TypeSafeError as exc:  # e.g. missing TYPESAFE_API_KEY
        print(f"error: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(result.to_dict(), indent=2))
    return _EXIT[result.verdict]


if __name__ == "__main__":
    raise SystemExit(main())
