"""python -m pytest nlp/test_context.py   (no Whisper model needed)"""

import functools
import json

import pytest

from nlp import context, enhance as enh, label, vocab
from nlp.context import extract_terms, set_context

SLIDES = """Lecture 3: Sampling Methods

Gibbs Sampling
- Gibbs sampling updates one parameter at a time from its full conditional
- Each full conditional distribution is easy to sample from
- Gibbs sampling is a special case of Metropolis-Hastings

Diagnostics
- Check convergence with the Gelman-Rubin statistic (R-hat)
- Gelman recommends running several chains
- Use the effective sample size (ESS) to judge mixing
- Low ESS means slow mixing; report ESS for every parameter

Homework 3 due Friday
"""


@pytest.fixture(autouse=True)
def small_vocab(tmp_path, monkeypatch):
    """A fixed vocab.json so the test doesn't depend on the real one."""
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps({"terms": ["Bayesian", "Metropolis-Hastings"]}))
    for f in (vocab.load_vocab, vocab.get_terms, vocab.build_hotwords, vocab.fix_terms,
              vocab.find_terms, vocab.highlight_terms):
        monkeypatch.setattr(f, "__defaults__", tuple(
            p if d == vocab.VOCAB_PATH else d for d in f.__defaults__))
    monkeypatch.setattr(context.load_vocab, "__defaults__", vocab.load_vocab.__defaults__)
    fixed = functools.partial(vocab.fix_terms, path=p)
    monkeypatch.setattr(enh, "fix_terms", fixed)
    monkeypatch.setattr(label, "fix_terms", fixed)
    for sid in ("lecture-3", "other"):
        context.clear_context(sid)


def test_extracts_phrases_names_and_acronyms():
    hints, fix = extract_terms(SLIDES)
    assert "full conditional" in hints
    assert "ESS" in hints          # acronym used several times
    assert "Gelman" in hints       # a name, capitalized mid-sentence
    assert "Gibbs sampling" in hints  # capital kept: never lowercase in the slides
    assert "full conditional" in fix and "ESS" in fix


def test_skips_noise_and_known_terms():
    hints, _ = extract_terms(SLIDES)
    lowered = [h.lower() for h in hints]
    assert "friday" not in lowered and "homework" not in lowered   # said once
    assert "metropolis-hastings" not in lowered                    # already in vocab.json
    assert not any(h.lower().startswith(("the ", "of ")) for h in hints)


def test_session_terms_fix_captions_for_that_session_only():
    set_context("lecture-3", SLIDES)
    cap = {"type": "caption", "session_id": "lecture-3", "segment_id": 1, "speaker": None,
           "text": "each full conditionel is easy", "start": 0.0, "end": 2.0, "is_final": True}
    assert enh.enhance(cap)["text"] == "each full conditional is easy"
    other = dict(cap, session_id="other")
    assert enh.enhance(other)["text"] == "each full conditionel is easy"


def test_same_upload_twice_is_ignored_and_new_text_adds():
    set_context("lecture-3", SLIDES)
    v = context.version("lecture-3")
    set_context("lecture-3", SLIDES)                 # 30 students, same deck
    assert context.version("lecture-3") == v
    info = set_context("lecture-3", "Hamiltonian dynamics\nHamiltonian dynamics again")
    assert context.version("lecture-3") == v + 1
    assert "Hamiltonian dynamics" in info["hints"] and "ESS" in info["hints"]


def test_hotwords_include_slide_terms_after_course_terms():
    set_context("lecture-3", SLIDES)
    hw = vocab.build_hotwords(extra=context.hint_terms("lecture-3"))
    assert hw.index("Bayesian") < hw.index("full conditional")


def test_no_slides_means_no_change():
    assert context.hint_terms("nobody") == () and context.fix_terms_for(None) == ()
