"""Evaluate the detector on a labelled JSONL file.

Each line: {"text": "...", "label": 1|0, "source": "optional", "context": "optional"}
(label 1 = injection). Reports AUROC on the risk score and confusion counts for
"flagged" (verdict != allow) and "blocked" (verdict == block).

Thresholds matter: detection quality is regime-dependent, so calibrate the
review/block thresholds on data that looks like your own traffic.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

from jev_pi_detect.detector import Detection, DetectorConfig, PromptInjectionDetector, detect_offline


def auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def confusion(labels: list[int], predicted: list[bool]) -> dict[str, float]:
    tp = sum(1 for y, p in zip(labels, predicted) if y and p)
    fp = sum(1 for y, p in zip(labels, predicted) if not y and p)
    fn = sum(1 for y, p in zip(labels, predicted) if y and not p)
    tn = sum(1 for y, p in zip(labels, predicted) if not y and not p)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
    }


def summarize(labels: list[int], results: list[Detection]) -> dict[str, object]:
    return {
        "n": len(labels),
        "errors": sum(1 for r in results if r.error),
        "auroc_risk": auroc(labels, [r.risk for r in results]),
        "auroc_jev_only": auroc(labels, [r.jev_risk or 0.0 for r in results]) if any(r.jev_risk is not None for r in results) else None,
        "auroc_heuristic_only": auroc(labels, [r.heuristic_risk for r in results]),
        "flagged": confusion(labels, [r.verdict != "allow" for r in results]),
        "blocked": confusion(labels, [r.verdict == "block" for r in results]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-pi-detect eval")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--block", type=float, default=0.8)
    parser.add_argument("--review", type=float, default=0.4)
    parser.add_argument("--details", action="store_true", help="print one line per example")
    parser.add_argument("--offline", action="store_true", help="heuristics only, no Jev calls (no API key needed)")
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    config = DetectorConfig(model=args.model, block_threshold=args.block, review_threshold=args.review)
    results: list[Detection] = []
    with nullcontext() if args.offline else PromptInjectionDetector(config=config) as detector:
        for row in rows:
            if detector is None:
                r = detect_offline(row["text"], config)
            else:
                r = detector.detect(row["text"], context=row.get("context"), source=row.get("source"))
            results.append(r)
            if args.details:
                print(f"{row['label']} {r.verdict:6} {r.risk:.2f} {r.technique or '-':22} {row['text'][:70]!r}", file=sys.stderr)
    print(json.dumps(summarize([int(r["label"]) for r in rows], results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
