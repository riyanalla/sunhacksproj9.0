"""python -m pytest nlp/test_enhance.py

Uses a fixed test vocab so it doesn't break when you edit vocab.json.
"""

import functools
import json

import pytest

from nlp import enhance as enh
from nlp import label
from nlp.vocab import fix_terms

VOCAB = {"terms": ["Gibbs sampling", "Markov Chain Monte Carlo", "Bayesian"]}


@pytest.fixture(autouse=True)
def test_vocab(tmp_path, monkeypatch):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(VOCAB))
    fixed = functools.partial(fix_terms, path=p)
    monkeypatch.setattr(enh, "fix_terms", fixed)
    monkeypatch.setattr(label, "fix_terms", fixed)


def cap(text, start, end, final=True, seg=1, **extra):
    return {"type": "caption", "session_id": "lecture-1", "segment_id": seg,
            "speaker": None, "text": text, "start": start, "end": end,
            "is_final": final, **extra}


@pytest.fixture(autouse=True)
def fresh():
    enh.setup("pause")
    label.reset()


def test_fixes_terms_and_keeps_other_fields():
    c = cap("we use gibs sampling", 1.0, 3.0, extra_field="kept")
    out = enh.enhance(c)
    assert out["text"] == "we use Gibbs sampling"
    assert out["speaker"] == "Speaker 1"
    assert {k: v for k, v in out.items() if k not in ("text", "speaker")} == \
           {k: v for k, v in c.items() if k not in ("text", "speaker")}


def test_does_not_modify_input():
    c = cap("basian stats", 0.0, 1.0)
    enh.enhance(c)
    assert c["text"] == "basian stats" and c["speaker"] is None


def test_long_pause_switches_speaker():
    assert enh.enhance(cap("first", 0.0, 2.0, seg=1))["speaker"] == "Speaker 1"
    assert enh.enhance(cap("still me", 2.3, 4.0, seg=2))["speaker"] == "Speaker 1"
    assert enh.enhance(cap("a question", 7.0, 9.0, seg=3))["speaker"] == "Speaker 2"


def test_partials_predict_speaker_without_changing_history():
    enh.enhance(cap("first", 0.0, 2.0, seg=1))
    # Draft of the next segment after a long pause: shows the speaker it will get...
    draft = enh.enhance(cap("a ques", 7.0, 8.0, final=False, seg=2))
    assert draft["speaker"] == "Speaker 2"
    # ...but many drafts don't advance the labels.
    for _ in range(3):
        enh.enhance(cap("a ques", 7.0, 8.0, final=False, seg=2))
    assert enh.enhance(cap("a question", 7.0, 9.0, seg=2))["speaker"] == "Speaker 2"


def test_partial_text_is_fixed_too():
    assert enh.enhance(cap("the markov chain monte carlo", 0, 1, final=False))["text"] == \
           "the Markov Chain Monte Carlo"


def test_session_start_resets_speakers():
    enh.enhance(cap("first", 0.0, 2.0))
    enh.enhance(cap("second", 7.0, 9.0))
    msg = {"type": "session_start", "session_id": "lecture-2"}
    assert enh.enhance(msg) == msg
    assert enh.enhance(cap("new lecture", 100.0, 101.0))["speaker"] == "Speaker 1"


def test_never_raises_on_bad_input():
    assert enh.enhance({"type": "caption", "text": None})["text"] == ""
    assert enh.enhance({"type": "session_end", "session_id": "x"})["type"] == "session_end"
    assert enh.enhance({})["text"] == ""
