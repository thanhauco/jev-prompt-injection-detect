"""Deterministic signals that run alongside Jev.

These are not a detector on their own. They exist so that a single steered or
evaded Jev call cannot by itself wave through text that carries well-known
injection markers, and so the patterns that Jev is known to be steered by
(counterfeit policy blocks, notes addressed to the classifier) are flagged
explicitly.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass

# Letter-repeat tolerant fragments: "ignnore", "instrructions".
_IGNORE = r"ig+n+o+r+e+|disregard|forget|override|bypass|skip"
_PRIOR = r"(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier|preceding|original|system|initial)"
_INSTR = r"in+s+t+r+u+c+t+i+o+n+s?|prompts?|rules|guidelines|directives|context|messages?"
_JUDGE = r"(?:ai|llm|model|classifier|assistant|agent|reviewer|moderator|judge|filter|scanner|detector)s?\b"


@dataclass(frozen=True)
class Signal:
    name: str
    weight: float
    match: str


_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    ("override_instructions", 0.9, re.compile(rf"\b(?:{_IGNORE})\b[^.\n]{{0,40}}\b{_PRIOR}\b[^.\n]{{0,20}}\b(?:{_INSTR})\b", re.I)),
    ("new_instructions", 0.6, re.compile(r"\b(?:new|updated|real|actual|true)\s+(?:system\s+)?(?:instructions|task|objective|rules)\s*(?:are|is|:)", re.I)),
    ("role_reassignment", 0.6, re.compile(r"\byou\s+are\s+now\s+(?:an?\s+|in\s+|the\s+|my\s+)?(?:[\w-]+\s+)?(?:assistant|ai|mode|persona|character|model|bot|dan)\b|\byou\s+are\s+no\s+longer\s+(?:bound|restricted|limited|required|an?\s+ai)\b|\bfrom\s+now\s+on,?\s+you\b|\bact\s+as\s+(?:an?\s+)?(?:unfiltered|unrestricted|jailbroken)", re.I)),
    ("jailbreak_persona", 0.7, re.compile(r"\b(?:DAN|developer\s+mode|god\s+mode|jailbr(?:ea|o)k(?:en)?|do\s+anything\s+now)\b", re.I)),
    ("system_prompt_extraction", 0.7, re.compile(r"\b(?:reveal|print|repeat|output|show|leak|dump)\b[^.\n]{0,30}\b(?:system\s+prompt|hidden\s+instructions|initial\s+prompt|your\s+instructions)\b", re.I)),
    ("fake_role_marker", 0.6, re.compile(r"(?:^|\n)\s*(?:\[/?(?:system|inst|admin|developer)\]|<\|?(?:im_start|system|endoftext)\|?>|#{2,}\s*(?:system|admin)\b|(?:system|assistant)\s*:)", re.I)),
    ("exfiltration", 0.7, re.compile(r"\b(?:send|forward|email|post|upload|exfiltrate)\b[^.\n]{0,60}\b(?:password|api\s*key|token|credentials?|secrets?|conversation|chat\s+history)\b|!\[[^\]]*\]\(https?://[^)]*\?[^)]*=", re.I)),
    ("tool_hijack", 0.5, re.compile(r"\b(?:call|invoke|run|execute|use)\s+(?:the\s+)?(?:tool|function|command|shell)\b[^.\n]{0,40}\b(?:immediately|now|without\s+(?:asking|confirmation))", re.I)),
    # Counterfeit policy / notes addressed to the judge: the steering vector reported against Jev.
    ("classifier_address", 0.8, re.compile(
        rf"\b(?:note|message|instruction|guidance|directive)s?\s+(?:to|for)\s+(?:the\s+|any\s+|all\s+)?(?:automated\s+)?{_JUDGE}"
        rf"|\bto\s+(?:any|all|the)\s+(?:ai|llm|automated)\s+(?:assistant|agent|model|system|reviewer)s?\b"
        rf"|\b{_JUDGE}\s*:\s*(?:this|the)\b", re.I)),
    ("verdict_injection", 0.7, re.compile(r"\b(?:label|classification|verdict|classify\s+(?:this\s+|it\s+)?as|mark\s+(?:this\s+|it\s+)?as)\s*:?\s*(?:benign|safe|clean|harmless|not\s+malicious)\b", re.I)),
    ("safety_assertion", 0.7, re.compile(r"\b(?:verified|approved|certified|confirmed)\s+(?:as\s+)?(?:safe|benign|not\s+(?:an?\s+)?(?:prompt\s+)?injection)|\bpre-?approved\b|\bwhitelisted\b|\bnot\s+(?:an?\s+)?(?:prompt\s+)?injection\b", re.I)),
    # Plain "this content is safe" is common in real documents, so on its own it is weak.
    ("safety_claim", 0.35, re.compile(r"\b(?:this|the\s+following)\s+(?:content|text|message|document|email|input)\s+is\s+(?:safe|benign|harmless)\b", re.I)),
    ("counterfeit_policy", 0.7, re.compile(r"\b(?:security|safety|content|moderation|updated?)\s+polic(?:y|ies)\b[^\n]{0,80}\b(?:exempt|override|supersede|classify|label|mark\s+(?:as|content)|scanners?|filters?)", re.I)),
    ("hidden_markup", 0.5, re.compile(r"<!--[\s\S]{0,500}?(?:ignore|instruction|assistant|system|ai)[\s\S]{0,500}?-->|style\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|color\s*:\s*(?:white|#fff\b))", re.I)),
]

_B64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_B64_SUSPICIOUS = re.compile(r"ignore|instruction|system|prompt|assistant", re.I)


def _decoded_base64_signals(text: str) -> list[Signal]:
    signals = []
    for blob in _B64.findall(text):
        try:
            decoded = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if _B64_SUSPICIOUS.search(decoded):
            signals.append(Signal("base64_payload", 0.7, decoded[:80]))
    return signals


def scan(text: str, obfuscation: list[str] | None = None, raw: str | None = None) -> list[Signal]:
    """Return heuristic signals found in (already normalized) ``text``.

    ``raw`` is the original text; encoded payloads are decoded from it because normalization
    would corrupt them.
    """
    signals = [Signal(name, weight, m.group(0)[:120]) for name, weight, pattern in _PATTERNS if (m := pattern.search(text))]
    signals.extend(_decoded_base64_signals(raw if raw is not None else text))
    # Evasion is only worth flagging when it is hiding something.
    if obfuscation and signals:
        signals.append(Signal("obfuscation", 0.3, ",".join(obfuscation)))
    return signals


def heuristic_score(signals: list[Signal]) -> float:
    """Noisy-OR combination of signal weights, in [0, 1]."""
    p_clean = 1.0
    for s in signals:
        p_clean *= 1.0 - s.weight
    return 1.0 - p_clean
