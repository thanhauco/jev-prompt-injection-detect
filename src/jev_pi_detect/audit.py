"""Audit log and review queue.

Every screening decision is appended to a JSONL file so you can measure the
filter on real traffic, investigate incidents, and work through the "review"
queue. Raw text is *not* stored by default (only a hash and length), because
the content is untrusted and often private; opt in with ``store_preview``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jev_pi_detect.detector import Detection


@dataclass
class AuditLog:
    path: Path
    store_preview: int = 0
    """Characters of the text to keep in each record; 0 stores none."""

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, text: str, detection: Detection, *, source: str = "unknown") -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": time.time(),
            "source": source,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text),
            "verdict": detection.verdict,
            "risk": round(detection.risk, 4),
            "jev_risk": None if detection.jev_risk is None else round(detection.jev_risk, 4),
            "heuristic_risk": round(detection.heuristic_risk, 4),
            "technique": detection.technique,
            "reasons": detection.reasons,
            "error": detection.error,
            "label": None,
        }
        if self.store_preview:
            entry["preview"] = text[: self.store_preview]
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def review_queue(self) -> list[dict[str, Any]]:
        """Unlabelled records the filter was unsure about, or that failed, newest first."""
        pending = [e for e in self.entries() if e["label"] is None and (e["verdict"] == "review" or e["error"])]
        return sorted(pending, key=lambda e: e["ts"], reverse=True)

    def label(self, sha256: str, is_injection: bool) -> int:
        """Record a human decision for every entry with this content hash; returns how many were updated."""
        entries = self.entries()
        n = 0
        for e in entries:
            if e["sha256"] == sha256:
                e["label"] = int(is_injection)
                n += 1
        if n:
            self.path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
        return n

    def summary(self) -> dict[str, Any]:
        entries = self.entries()
        verdicts = Counter(e["verdict"] for e in entries)
        return {
            "total": len(entries),
            "verdicts": dict(verdicts),
            "errors": sum(1 for e in entries if e["error"]),
            "pending_review": len(self.review_queue()),
            "by_source": dict(Counter(e["source"] for e in entries)),
            "by_technique": dict(Counter(e["technique"] for e in entries if e["verdict"] != "allow" and e["technique"])),
        }

    def labelled_dataset(self, texts: dict[str, str]) -> list[dict[str, Any]]:
        """Human-labelled records as eval rows, given a hash -> text mapping you kept elsewhere."""
        return [{"text": texts[e["sha256"]], "label": e["label"], "source": e["source"]} for e in self.entries() if e["label"] is not None and e["sha256"] in texts]
