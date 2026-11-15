"""Word error rate (WER) between what you meant to say and what was transcribed.

    r = score("Submit your assignments by Friday.", "submit your airplanes by friday")
    r.wer        # 0.2  (1 wrong word out of 5)
    r.errors     # [Error(said='assignments', heard='airplanes')]

Text is normalized first so formatting differences don't count as mistakes:
case, punctuation, "10%" vs "ten percent", "OK" vs "okay", "4th" vs "fourth".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen "
         "fourteen fifteen sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}
SAME = {"ok": "okay", "alright": "all right", "till": "until", "gonna": "going to",
        "wanna": "want to", "percentage": "percent"}


def _num_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if not rest else " " + _num_words(rest))
    if n < 1_000_000:
        rest = n % 1000
        return _num_words(n // 1000) + " thousand" + ("" if not rest else " " + _num_words(rest))
    return str(n)


def _ordinal(n: int) -> str:
    words = _num_words(n).split()
    last = words[-1]
    if last in _ORD:
        words[-1] = _ORD[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


def _expand_number(m: re.Match) -> str:
    digits, decimals, suffix = m.group(1).replace(",", ""), m.group(2), (m.group(3) or "").lower()
    n = int(digits)
    if suffix in ("st", "nd", "rd", "th"):
        return " " + _ordinal(n) + " "
    out = _num_words(n)
    if decimals:
        out += " point " + " ".join(_ONES[int(d)] for d in decimals)
    return " " + out + " "


def normalize(text: str) -> list[str]:
    """Lowercase words with formatting differences removed."""
    t = text.lower()
    t = t.replace("%", " percent ").replace("&", " and ")
    t = re.sub(r"(\d[\d,]*)(?:\.(\d+))?(st|nd|rd|th)?", _expand_number, t)
    t = re.sub(r"[-/]", " ", t)
    t = re.sub(r"[^a-z0-9' ]", " ", t)
    words = []
    for w in t.split():
        w = w.strip("'")
        if w:
            words += SAME.get(w, w).split()
    return words


@dataclass
class Error:
    said: str    # words in the reference ("" if Whisper added words)
    heard: str   # words in the transcript ("" if Whisper dropped words)

    def __str__(self) -> str:
        if not self.heard:
            return f'missed   "{self.said}"'
        if not self.said:
            return f'added    "{self.heard}"'
        return f'"{self.said}"  ->  heard "{self.heard}"'


@dataclass
class Result:
    ref_words: int
    substitutions: int
    deletions: int
    insertions: int
    errors: list[Error] = field(default_factory=list)
    diff: str = ""  # reference with errors marked inline

    @property
    def wer(self) -> float:
        return (self.substitutions + self.deletions + self.insertions) / max(1, self.ref_words)


def _align(ref: list[str], hyp: list[str]) -> list[tuple[str, str | None, str | None]]:
    """Minimum-edit alignment: list of (op, ref_word, hyp_word), op in = sub del ins.

    Among equally cheap alignments, prefer the one that matches the most words, so
    "is due today" vs "due today today" reads as missed "is" + added "today".
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = (edits, -matches) for ref[:i] vs hyp[:j]
    d = [[(0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = (i, 0)
    for j in range(m + 1):
        d[0][j] = (j, 0)

    def diag(i, j):
        e, nm = d[i - 1][j - 1]
        return (e, nm - 1) if ref[i - 1] == hyp[j - 1] else (e + 1, nm)

    def up(i, j):
        return (d[i - 1][j][0] + 1, d[i - 1][j][1])

    def left(i, j):
        return (d[i][j - 1][0] + 1, d[i][j - 1][1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(diag(i, j), up(i, j), left(i, j))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == diag(i, j):
            ops.append(("=" if ref[i - 1] == hyp[j - 1] else "sub", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and (not j or d[i][j] == up(i, j)):
            ops.append(("del", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, hyp[j - 1]))
            j -= 1
    return ops[::-1]


def score(reference: str, transcript: str) -> Result:
    ref, hyp = normalize(reference), normalize(transcript)
    ops = _align(ref, hyp)
    r = Result(len(ref), *(sum(op == k for op, _, _ in ops) for k in ("sub", "del", "ins")))

    # Group neighbouring mistakes into one readable error ("ten percent" -> "tenth 1.5").
    parts, said, heard = [], [], []

    def close():
        if said or heard:
            e = Error(" ".join(said), " ".join(heard))
            r.errors.append(e)
            parts.append(f"[{e.said or '+'} -> {e.heard or '-'}]")
            said.clear()
            heard.clear()

    for op, a, b in ops:
        if op == "=":
            close()
            parts.append(a)
        else:
            if a:
                said.append(a)
            if b:
                heard.append(b)
    close()
    r.diff = " ".join(parts)
    return r
