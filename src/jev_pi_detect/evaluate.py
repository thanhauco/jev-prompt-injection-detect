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
import math
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from statistics import NormalDist

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


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n; stays valid near 0 and 1 and for small n."""
    if n == 0:
        return (0.0, 1.0)
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def exact_lower(k: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact (Clopper-Pearson) lower bound on a rate after k successes in n trials.

    Smallest p for which seeing >= k successes still has probability >= 1 - confidence; no
    normal approximation, so it is the right test for "is the rate provably >= target".
    """
    if n == 0 or k == 0:
        return 0.0
    alpha = 1 - confidence
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        tail = sum(math.comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(k, n + 1))  # P(X >= k | mid)
        lo, hi = (mid, hi) if tail < alpha else (lo, mid)
    return hi


def min_samples_zero_misses(target: float = 0.95, confidence: float = 0.95) -> int:
    """Smallest n such that n/n correct proves rate >= target at the given confidence (59 for 95%/95%)."""
    return math.ceil(math.log(1 - confidence) / math.log(target))


def bootstrap_auroc(labels: list[int], scores: list[float], confidence: float = 0.95, iters: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    if auroc(labels, scores) is None:
        return None
    rng = random.Random(seed)
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    vals = []
    for _ in range(iters):
        bp = [rng.choice(pos) for _ in pos]
        bn = [rng.choice(neg) for _ in neg]
        vals.append(auroc([1] * len(bp) + [0] * len(bn), bp + bn))
    vals.sort()
    lo = vals[int(((1 - confidence) / 2) * iters)]
    hi = vals[min(iters - 1, int((1 - (1 - confidence) / 2) * iters))]
    return (lo, hi)


def confidence_report(conf: dict[str, float], labels: list[int], scores: list[float], target: float, confidence: float) -> dict[str, object]:
    """Does the detector *provably* reach `target` recall and specificity at `confidence`? (exact one-sided lower bound >= target)"""
    pos, neg = conf["tp"] + conf["fn"], conf["tn"] + conf["fp"]
    r_lo, r_hi = wilson(int(conf["tp"]), int(pos), confidence)
    s_lo, s_hi = wilson(int(conf["tn"]), int(neg), confidence)
    r_exact, s_exact = exact_lower(int(conf["tp"]), int(pos), confidence), exact_lower(int(conf["tn"]), int(neg), confidence)
    auc_ci = bootstrap_auroc(labels, scores, confidence)
    return {
        "target": target, "confidence": confidence,
        "recall": {"value": conf["recall"], "ci": [r_lo, r_hi], "n": pos, "exact_lower_bound": r_exact, "proven": r_exact >= target},
        "specificity": {"value": 1 - conf["false_positive_rate"], "ci": [s_lo, s_hi], "n": neg, "exact_lower_bound": s_exact, "proven": s_exact >= target},
        "auroc_ci": list(auc_ci) if auc_ci else None,
        "min_samples_per_class_if_no_errors": min_samples_zero_misses(target, confidence),
    }


def summarize(labels: list[int], results: list[Detection], target: float = 0.95, confidence: float = 0.95) -> dict[str, object]:
    flagged = confusion(labels, [r.verdict != "allow" for r in results])
    return {
        "confidence_report_flagged": confidence_report(flagged, labels, [r.risk for r in results], target, confidence),
        "n": len(labels),
        "errors": sum(1 for r in results if r.error),
        "auroc_risk": auroc(labels, [r.risk for r in results]),
        "auroc_jev_only": auroc(labels, [r.jev_risk or 0.0 for r in results]) if any(r.jev_risk is not None for r in results) else None,
        "auroc_heuristic_only": auroc(labels, [r.heuristic_risk for r in results]),
        "flagged": flagged,
        "blocked": confusion(labels, [r.verdict == "block" for r in results]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jev-pi-detect eval")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default=None)
    parser.add_argument("--block", type=float, default=0.8)
    parser.add_argument("--review", type=float, default=0.4)
    parser.add_argument("--details", action="store_true", help="print one line per example")
    parser.add_argument("--target", type=float, default=0.95, help="required recall/specificity (default 0.95)")
    parser.add_argument("--confidence", type=float, default=0.95, help="confidence level for intervals (default 0.95)")
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
    print(json.dumps(summarize([int(r["label"]) for r in rows], results, args.target, args.confidence), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
