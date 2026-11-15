"""Lecture context from uploaded slides / notes: pull out the key terms per session.

    from nlp.context import set_context

    info = set_context("lecture-1", slides_text)   # server: after a student uploads
    # -> {"hints": [...], "fix": [...]}   (what was extracted, for logging / the UI)

After this, captions for "lecture-1" get those terms automatically: `enhance()` corrects
near-misses of the "fix" terms, and a `Transcriber(session_id="lecture-1")` sends the
"hints" to Whisper. It takes effect from the next chunk; nothing restarts.

Whisper only reads ~220 tokens of hints, so the whole deck can't go in. Instead:
- hints: up to MAX_HINTS terms that make Whisper more likely to hear them: repeated
  phrases ("posterior distribution"), names ("Gelman"), acronyms ("BDA"), and long words
  that come up repeatedly ("likelihood").
- fix: the subset safe to *correct the text toward*: phrases, names and acronyms, never
  ordinary single words, so a near-miss of "posterior distribution" is fixed but a
  normal word is never turned into a slide word.
Terms already in vocab.json are skipped (they're used anyway).

Uploading more files for the same session adds to its context; uploading the same text
twice (30 students, one deck) is ignored.
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .vocab import COMMON_WORDS, load_vocab

MAX_HINTS = 30  # leaves room for vocab.json within Whisper's hint budget
MAX_FIX = 20
MAX_NGRAM = 4

# Words that can't start or end a key phrase (on top of vocab.COMMON_WORDS).
STOP = COMMON_WORDS | set("""
an the and or of to in on at by for with from as is are was were be been this that these
those it its we you they i our their your his her via vs per eg ie etc al et using based
slide slides lecture chapter section page figure fig table example examples note notes
week today tomorrow recap review outline overview introduction summary reading due
""".split())
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*[A-Za-z]|[A-Za-z]")
_PIECE_BREAK = re.compile(r"[^A-Za-z'\- \t]+")  # punctuation, digits, newlines, bullets


@dataclass
class _Session:
    texts: list[str] = field(default_factory=list)
    hashes: set[str] = field(default_factory=set)
    hints: tuple[str, ...] = ()
    fix: tuple[str, ...] = ()
    version: int = 0


_sessions: dict[str, _Session] = {}
_lock = threading.Lock()


def set_context(session_id: str, text: str) -> dict:
    """Add slide/notes text to a session. Returns the terms now in use."""
    text = text or ""
    digest = hashlib.sha1(text.strip().encode("utf-8", "ignore")).hexdigest()
    with _lock:
        s = _sessions.setdefault(str(session_id), _Session())
        if digest not in s.hashes and text.strip():
            s.hashes.add(digest)
            s.texts.append(text)
            s.hints, s.fix = extract_terms("\n".join(s.texts))
            s.version += 1
        return {"hints": list(s.hints), "fix": list(s.fix)}


def clear_context(session_id: str) -> None:
    with _lock:
        _sessions.pop(str(session_id), None)


def _get(session_id) -> _Session | None:
    return _sessions.get(str(session_id)) if session_id is not None else None


def hint_terms(session_id) -> tuple[str, ...]:
    s = _get(session_id)
    return s.hints if s else ()


def fix_terms_for(session_id) -> tuple[str, ...]:
    s = _get(session_id)
    return s.fix if s else ()


def version(session_id) -> int:
    """Changes whenever the session's terms change (for caching)."""
    s = _get(session_id)
    return s.version if s else 0


def read_slides(path: str | Path) -> str:
    """Text of a .txt/.md file or a .pdf (needs pypdf). For tests and local tools;
    the server extracts text itself and calls set_context with it."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


# --- term extraction --------------------------------------------------------------


def _key(term: str) -> str:
    """Comparison form: "Bayes' theorem" == "bayes theorem", "parameters" == "parameter"."""
    words = re.findall(r"[a-z0-9]+", term.lower().replace("'", ""))
    return " ".join(w[:-1] if len(w) > 4 and w.endswith("s") and not w.endswith("ss")
                    else w for w in words)


def extract_terms(text: str, max_hints: int = MAX_HINTS,
                  max_fix: int = MAX_FIX) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """(hints, fix) key terms of a lecture's slides/notes, most important first."""
    known = {_key(t) for k in ("terms", "acronyms", "professor") for t in load_vocab()[k]}

    phrase_counts: Counter[str] = Counter()  # lowercase phrase -> occurrences
    spellings: dict[str, Counter[str]] = {}  # lowercase phrase -> original spellings
    acronyms: Counter[str] = Counter()
    names: Counter[str] = Counter()          # capitalized uses (outside Title Case lines)
    mid_caps: set[str] = set()               # ...at least one of them mid-sentence
    lowercase_seen: set[str] = set()
    long_words: Counter[str] = Counter()

    # Phrases never run across punctuation or numbers: "interpretation: probability"
    # is two pieces. The first word of a piece may be capitalized just for starting.
    for piece in _PIECE_BREAK.split(text):
        words = _WORD.findall(piece)
        # Title Case ("Two Interpretations of Probability"): capitals don't mean names.
        title = len(words) >= 2 and sum(w[0].isupper() for w in words) >= 0.6 * len(words)
        for i, w in enumerate(words):
            lw = w.lower()
            if w.isupper() and 2 <= len(w) <= 6:
                acronyms[w] += 1
                continue
            for part in w.split("-"):  # "Gelman-Rubin" also counts as "Gelman"
                if part[:1].isupper() and not title and part.lower() not in STOP:
                    names[part] += 1
                    if i > 0:
                        mid_caps.add(part.lower())
            if w[0].islower():
                lowercase_seen.add(lw)
            if len(lw) >= 8 and lw not in STOP:
                long_words[lw] += 1

        lower = [w.lower() for w in words]
        for n in range(2, MAX_NGRAM + 1):
            for i in range(len(words) - n + 1):
                gram = lower[i:i + n]
                if gram[0] in STOP or gram[-1] in STOP or any(len(g) < 2 for g in gram):
                    continue
                # Inside a phrase only "of" / "and" are allowed as small words.
                if any(g in STOP and g not in ("of", "and") for g in gram[1:-1]):
                    continue
                key = " ".join(gram)
                phrase_counts[key] += 1
                spellings.setdefault(key, Counter())[" ".join(words[i:i + n])] += 1

    # Phrases said at least twice; a longer phrase replaces its own pieces.
    phrases = {p: c for p, c in phrase_counts.items() if c >= 2}
    for p in list(phrases):
        if any(q != p and f" {p} " in f" {q} " and phrases[q] >= 0.7 * phrases[p]
               for q in phrases):
            del phrases[p]

    # A name is capitalized mid-sentence at least once, and never written in lowercase.
    name_set = {n.lower() for n in names if n.lower() in mid_caps
                and n.lower() not in lowercase_seen}
    capitalized = {n.lower() for n in names}

    def spelling(p: str) -> str:
        """Most common original spelling. A word keeps its capital if the slides write it
        capitalized outside titles and never in lowercase ("Gibbs sampling"); otherwise
        it's lowercased ("Bayesian Statistics" in a title -> "Bayesian statistics")."""
        best = spellings[p].most_common(1)[0][0]
        return " ".join(w if w.isupper() or (w.lower() in capitalized and w.lower() not in
                                             lowercase_seen) else w.lower() for w in best.split())

    scored: list[tuple[float, str, bool]] = []  # (score, term, safe to fix toward)
    for p, c in phrases.items():
        scored.append((c * len(p.split()), spelling(p), True))
    # Weak hints make Whisper worse (tested: a course code and "Friday" from the last
    # slide added errors), so names and acronyms must come up at least twice, and single
    # ordinary words need 3+ uses and a long, unusual-looking word.
    for a, c in acronyms.items():
        if c >= 2:
            scored.append((c * 2 + 1, a, True))
    for n, c in names.items():
        if n.lower() in name_set and len(n) >= 4 and c >= 2:
            scored.append((c * 1.5, n, True))
    for w, c in long_words.items():
        if c >= 3 and w not in name_set and len(w) >= 10:
            scored.append((c, w, False))

    hints, fix, seen = [], [], set()
    for _score, term, safe in sorted(scored, key=lambda x: (-x[0], x[1])):
        k = _key(term)
        if not k or k in seen or k in known:
            continue
        # A single word that's part of a chosen or vocab.json phrase adds nothing.
        if " " not in k and any(k in h.split() for h in seen | known):
            continue
        seen.add(k)
        hints.append(term)
        if safe and len(fix) < max_fix:
            fix.append(term)
        if len(hints) >= max_hints:
            break
    return tuple(hints), tuple(fix)
