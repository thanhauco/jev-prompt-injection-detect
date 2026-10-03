"""Text normalization that undoes common classifier-evasion tricks.

Token-breaking ("i.g.n.o.r.e", "ignnore", zero-width joiners, homoglyphs) is the
cheapest way to slip an injection past a classifier. We send Jev both the raw
text and a normalized view, and run heuristics on the normalized view.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Invisible characters commonly used to split tokens or hide text.
_INVISIBLE = re.compile(
    "[­͏؜ᅟᅠ឴឵᠎​-‏‪-‮"
    "⁠-⁤⁦-⁯ㅤ﻿ﾠ\U000e0000-\U000e007f]"
)

# Cyrillic/Greek letters that render like Latin ones (NFKC leaves these alone).
_HOMOGLYPHS = str.maketrans(
    {
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l",
        "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
        "Р": "P", "С": "C", "Т": "T", "Х": "X", "Ү": "Y",
        "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "Ι": "I", "Ο": "O",
    }
)

# Single letters separated by punctuation/spaces: "i g n o r e", "s.y.s.t.e.m".
_SPACED_LETTERS = re.compile(r"\b(?:[A-Za-z][\s.\-_*|/\\]{1,3}){3,}[A-Za-z]\b")
_SEPARATORS = re.compile(r"[\s.\-_*|/\\]+")
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_LEET_WORD = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[013457@$])[\w@$]{3,}\b")
_LEET_SANDWICH = re.compile(r"[A-Za-z][013457@$]+[A-Za-z]")
_LEET_LEADING = re.compile(r"[013457@$][A-Za-z]{2}")
# Runs of 3+ of the same letter ("iiiignore") collapse to one.
_REPEATS = re.compile(r"([A-Za-z])\1{2,}")


def _join_spaced(m: re.Match[str]) -> str:
    run = m.group(0)
    # "i.g.n.o.r.e p.r.e.v" uses punctuation inside words, so whitespace still separates words;
    # "i g n o r e  a l l" uses wider gaps between words.
    if re.search(r"[^\sA-Za-z]", run):
        words = run.split()
    elif re.search(r"\s{2,}", run):
        words = re.split(r"\s{2,}", run)
    else:
        words = [run]
    return " ".join(_SEPARATORS.sub("", word) for word in words)


def _deleet(m: re.Match[str]) -> str:
    # Only rewrite tokens that look like disguised words ("1gn0r3", "pr0mpt"), not units or
    # identifiers ("200g", "3pm", "LLM01").
    word = m.group(0)
    letters = sum(c.isalpha() for c in word)
    if _LEET_SANDWICH.search(word) or (_LEET_LEADING.match(word) and letters >= len(word) - letters and len(word) >= 4):
        return word.translate(_LEET)
    return word


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    obfuscation: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.normalized != self.original


def normalize(text: str) -> NormalizedText:
    """Return a de-obfuscated view of ``text`` and the tricks that were found."""
    found: list[str] = []
    out = unicodedata.normalize("NFKC", text)
    if out != text:
        found.append("unicode_compatibility_chars")

    stripped = _INVISIBLE.sub("", out)
    if stripped != out:
        found.append("invisible_chars")
    out = stripped

    mapped = out.translate(_HOMOGLYPHS)
    if mapped != out:
        found.append("homoglyphs")
    out = mapped

    joined = _SPACED_LETTERS.sub(_join_spaced, out)
    if joined != out:
        found.append("spaced_letters")
    out = joined

    deleet = _LEET_WORD.sub(_deleet, out)
    if deleet != out:
        found.append("leetspeak")
    out = deleet

    collapsed = _REPEATS.sub(r"\1", out)
    if collapsed != out:
        found.append("repeated_letters")
    out = collapsed

    return NormalizedText(original=text, normalized=out, obfuscation=found)
