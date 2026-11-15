"""Course vocabulary: Whisper prompt, fuzzy term fixing, keyword highlighting."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rapidfuzz import fuzz

VOCAB_PATH = Path(__file__).with_name("vocab.json")
SCORE_CUTOFF = 85
MIN_FUZZY_LEN = 5  # never fuzzy-fix spans shorter than this (letters only)

_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# Common English words that must never be "fixed" into a vocab term.
COMMON_WORDS = set("""
a about above after again against all also an and another any are around as at
back be because been before being below between both but by came can come could
day did do does doing done down during each even every few first for from get
give go going good got great had has have having he her here him his how i if in
into is it its just know last let like little long look made make many may me
might more most much must my need never new next no not now of off old on once
one only or other our out over own part people place point put right said same
say see she should show since so some something still such take than that the
their them then there these they thing things think this those through time to
too two under up us use used very want was way we well went were what when where
which while who why will with without word work would year yes yet you your
able actually always answer because better call called case change course
different down early end example fact find form found general given going hand
having head help high however important keep kind large later least left less
line lot mean means number often open order problem question quite rather real
really several side small sort start state story sure system talk tell thank
thanks today together try turn understand until value whole why write wrong
""".split())


@dataclass(frozen=True)
class Term:
    canonical: str   # spelling to put in the caption
    norm: str        # lowercase, punctuation stripped, single-spaced
    compact: str     # norm without spaces
    nwords: int
    acronym: bool


def _normalize(s: str) -> str:
    return " ".join(_WORD_RE.findall(s.lower()))


def load_vocab(path: str | Path = VOCAB_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(data.get(k, [])) for k in ("professor", "acronyms", "terms", "classroom")}


@lru_cache(maxsize=256)
def _terms_for(path: str, extra: tuple[str, ...] = ()) -> tuple[Term, ...]:
    vocab = load_vocab(path)
    entries: list[tuple[str, bool]] = [(a, True) for a in vocab["acronyms"]]
    entries += [(t, False) for t in vocab["terms"]]
    # Session terms from slides (context.py): ALL-CAPS ones are acronyms.
    entries += [(t, t.isupper() and " " not in t) for t in extra]
    for name in vocab["professor"]:
        entries.append((name, False))
        # Also match the surname alone ("Professor Smith" -> "Smith").
        surname = _WORD_RE.findall(name)[-1:] or []
        if surname and len(surname[0]) >= MIN_FUZZY_LEN:
            entries.append((surname[0], False))
    terms = []
    for canonical, is_acr in entries:
        norm = _normalize(canonical)
        if norm:
            terms.append(Term(canonical, norm, norm.replace(" ", ""), len(norm.split()), is_acr))
    return tuple(terms)


def get_terms(path: str | Path = VOCAB_PATH, extra: tuple[str, ...] = ()) -> tuple[Term, ...]:
    """vocab.json terms, plus `extra` terms (e.g. from this lecture's slides)."""
    return _terms_for(str(Path(path).resolve()), tuple(extra))


def build_initial_prompt(path: str | Path = VOCAB_PATH) -> str:
    """Text for faster-whisper's `initial_prompt` to bias it toward course terms."""
    v = load_vocab(path)
    parts = []
    if v["professor"]:
        parts.append(f"Lecture by {', '.join(v['professor'])}.")
    if v["terms"]:
        parts.append(f"Topics: {', '.join(v['terms'])}.")
    if v["acronyms"]:
        parts.append(f"Acronyms: {', '.join(v['acronyms'])}.")
    return " ".join(parts)


def build_hotwords(path: str | Path = VOCAB_PATH, extra: tuple[str, ...] = ()) -> str:
    """Text for faster-whisper's `hotwords`: course terms plus everyday class words.

    Unlike `initial_prompt`, this is sent with every chunk, so `initial_prompt` stays
    free for the previous sentence (see live_test.py). `extra` (this lecture's slide
    terms) goes after the course terms and before the everyday classroom words, which
    matter least: Whisper drops whatever is past its ~220-token limit.
    """
    v = load_vocab(path)
    have = {w.lower() for w in v["professor"] + v["terms"] + v["acronyms"]}
    extra = [t for t in extra if t.lower() not in have]
    words = v["professor"] + v["terms"] + v["acronyms"] + extra + v["classroom"]
    return ", ".join(words) + "." if words else ""


def _score(words: list[str], term: Term) -> float:
    """Similarity of a run of transcript words to a term, 0-100."""
    cand_norm = " ".join(w.lower() for w in words)
    cand_compact = cand_norm.replace(" ", "")

    if term.acronym:
        # Acronyms are short, so fuzzy matching would wreck normal words.
        # Accept only "cpu" or spelled-out "c p u".
        if cand_compact != term.compact:
            return 0.0
        if len(words) > 1 and any(len(w) != 1 for w in words):
            return 0.0
        return 100.0

    if len(cand_compact) < MIN_FUZZY_LEN and cand_compact != term.compact:
        return 0.0
    if any(w.lower() in COMMON_WORDS for w in words) and cand_compact != term.compact:
        return 0.0

    if cand_compact != term.compact and any(any(c.isdigit() for c in w) for w in words):
        return 0.0  # "(2) Bayesian" must not fuzzy-merge into "Bayesian"

    if len(words) == term.nwords:
        return max(fuzz.ratio(cand_norm, term.norm), fuzz.ratio(cand_compact, term.compact))
    # One extra word: Whisper split a term ("back propagation"). But if the term is already
    # there whole with a separate word beside it ("x Bayesian"), leave that word alone.
    for sub in (words[1:], words[:-1]):
        if "".join(w.lower() for w in sub) == term.compact:
            return 0.0
    return fuzz.ratio(cand_compact, term.compact)


def _plural_suffix(span: list[str], term: Term) -> str:
    """'s'/'es' if the span is just the term made plural ("hierarchical models")."""
    compact = "".join(w.lower() for w in span)
    if compact.startswith(term.compact) and compact[len(term.compact):] in ("s", "es"):
        if not term.compact.endswith("s"):
            return span[-1][len(span[-1]) - len(compact) + len(term.compact):]
    return ""


def fix_terms(text: str, path: str | Path = VOCAB_PATH, cutoff: float = SCORE_CUTOFF,
              extra_terms: tuple[str, ...] = ()) -> str:
    """Replace near-miss spellings of vocab terms (and `extra_terms`) with the canonical
    spelling."""
    terms = get_terms(path, extra_terms)
    if not terms or not text:
        return text
    words = list(_WORD_RE.finditer(text))
    max_n = max(max(t.nwords + 1, len(t.compact) if t.acronym else 0) for t in terms)

    out: list[str] = []
    cursor = 0  # position in `text` up to which output has been written
    i = 0
    while i < len(words):
        best: tuple[float, int, Term] | None = None  # (score, span length, term)
        for n in range(1, min(max_n, len(words) - i) + 1):
            span = [m.group() for m in words[i:i + n]]
            for term in terms:
                spelled_out = term.acronym and n == len(term.compact)  # "c p u"
                if n not in (term.nwords, term.nwords + 1) and not spelled_out:
                    continue
                s = _score(span, term)
                if s < cutoff:
                    continue
                # Higher score wins; on a tie prefer the exact word count.
                key = (s, n == term.nwords)
                if best is None or key > (best[0], best[1] == best[2].nwords):
                    best = (s, n, term)
        if best is None:
            i += 1
            continue
        _, n, term = best
        start, end = words[i].start(), words[i + n - 1].end()
        out.append(text[cursor:start])
        # Keep plurals: "hierarchical models" stays plural, not "hierarchical model".
        out.append(term.canonical + _plural_suffix([m.group() for m in words[i:i + n]], term))
        cursor = end
        i += n
    out.append(text[cursor:])
    return "".join(out)


def find_terms(text: str, path: str | Path = VOCAB_PATH,
               extra_terms: tuple[str, ...] = ()) -> list[dict]:
    """Character spans of vocab terms in (already fixed) text, for bolding in the UI."""
    spans = []
    for term in sorted(get_terms(path, extra_terms), key=lambda t: -len(t.canonical)):
        pattern = r"(?<![A-Za-z0-9])" + re.escape(term.canonical) + r"(?![A-Za-z0-9])"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            if not any(m.start() < s["end"] and s["start"] < m.end() for s in spans):
                spans.append({"start": m.start(), "end": m.end(), "term": term.canonical})
    return sorted(spans, key=lambda s: s["start"])


def highlight_terms(text: str, open_tag: str = "**", close_tag: str = "**",
                    path: str | Path = VOCAB_PATH, extra_terms: tuple[str, ...] = ()) -> str:
    """Wrap vocab terms in markers, e.g. '**gradient descent**'."""
    out, cursor = [], 0
    for s in find_terms(text, path, extra_terms):
        out += [text[cursor:s["start"]], open_tag, text[s["start"]:s["end"]], close_tag]
        cursor = s["end"]
    out.append(text[cursor:])
    return "".join(out)
