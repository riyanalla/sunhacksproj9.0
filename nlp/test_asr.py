"""python -m pytest nlp/test_asr.py   (no Whisper model needed)"""

from types import SimpleNamespace

import numpy as np

from nlp.asr import clean_text, is_junk
from nlp.chunker import AutoGain


def test_drops_word_fragment_at_chunk_start():
    assert clean_text(" nd to view this probability", "interpretation.") == \
        "to view this probability"
    assert clean_text(" nd, codifies prior knowledge", "methods.") == "codifies prior knowledge"


def test_keeps_real_short_words():
    assert clean_text(" in the field", "a theory") == "in the field"
    assert clean_text(" a prior", "based on") == "a prior"
    assert clean_text(" I think", "so") == "I think"


def test_drops_words_repeated_across_boundary():
    assert clean_text(" Interpretation, which views", "the frequentist interpretation") == \
        "which views"
    assert clean_text(" of belief in an event", "a degree of belief") == "in an event"


def test_keeps_real_repeats_that_are_not_at_the_boundary():
    # "degree of belief" really is said twice in a row in the test text
    assert clean_text(" The degree of belief may be", "a degree of belief in an event.") == \
        "The degree of belief may be"


def seg(text="hello", no_speech=0.0, logprob=-0.2, ratio=1.0):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob,
                           compression_ratio=ratio)


def test_junk_filter():
    assert not is_junk(seg())
    assert is_junk(seg(no_speech=0.9, logprob=-1.1))            # silence
    assert is_junk(seg("the the the the the the", ratio=3.0))  # repetition loop
    assert is_junk(seg(logprob=-1.5))                           # noise
    assert is_junk(seg("Thanks for watching!", logprob=-0.7))   # Whisper hallucination
    assert not is_junk(seg("Thanks for watching!", logprob=-0.1))  # said clearly: keep


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def test_auto_gain_boosts_quiet_mic_and_leaves_loud_one():
    rng = np.random.default_rng(0)
    quiet, loud = AutoGain(), AutoGain()
    for i in range(200):  # 6 s: background hiss, with "speech" in every other 0.3 s
        on = (i // 10) % 2
        q = quiet(rng.normal(0, 0.003 if on else 0.0002, 480).astype(np.float32))
        l = loud(rng.normal(0, 0.1 if on else 0.005, 480).astype(np.float32))
    assert rms(q) > 0.015         # ~-50 dBFS speech boosted to about -34 dBFS
    assert loud.gain == 1.0       # already loud: untouched


def test_auto_gain_does_not_boost_silence_before_speech():
    rng = np.random.default_rng(0)
    g = AutoGain()
    for _ in range(100):  # 3 s of room noise only
        g(rng.normal(0, 0.0005, 480).astype(np.float32))
    assert g.gain == 1.0
