"""Run from the repo root: python -m pytest nlp

Uses a fixed test vocab so it doesn't break when you edit vocab.json.
"""

import json

import pytest

from nlp.vocab import build_initial_prompt, find_terms, fix_terms, highlight_terms

VOCAB = {
    "professor": ["Dr. Ramirez"],
    "acronyms": ["CPU", "API", "TCP"],
    "terms": ["gradient descent", "backpropagation", "Kubernetes", "eigenvector"],
}


@pytest.fixture
def vocab(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(VOCAB))
    return p


@pytest.mark.parametrize("heard, expected", [
    ("we use gradiant descent here", "we use gradient descent here"),
    ("then backpropogation updates weights", "then backpropagation updates weights"),
    ("then back propagation updates weights", "then backpropagation updates weights"),
    ("deploy it on kubernetis", "deploy it on Kubernetes"),
    ("compute the eigen vector", "compute the eigenvector"),
    ("the cpu is busy", "the CPU is busy"),
    ("call the a p i", "call the API"),
    ("as Dr. Ramires said", "as Dr. Ramirez said"),
    ("ask Ramires", "ask Ramirez"),
])
def test_fixes_near_misses(vocab, heard, expected):
    assert fix_terms(heard, vocab) == expected


@pytest.mark.parametrize("text", [
    "two eigenvectors and several CPUs",           # plurals stay plural
    "(2) eigenvector methods",                     # list numbers aren't merged into terms
    "step 3 backpropagation",
])
def test_leaves_correct_text_alone(vocab, text):
    assert fix_terms(text, vocab) == text


def test_fixes_misspelled_plural(vocab):
    assert fix_terms("compute the eigen vectors", vocab) == "compute the eigenvectors"


@pytest.mark.parametrize("sentence", [
    "a pie is on the table",
    "the cup is full",
    "it went down the hill at a great descent",
    "we waited for the bus to arrive",
    "the top speed of the car",
    "I need to take a break",
    "the radiant sun",
])
def test_leaves_common_words_alone(vocab, sentence):
    assert fix_terms(sentence, vocab) == sentence


def test_keeps_punctuation(vocab):
    assert fix_terms("So, gradiant descent! Right?", vocab) == "So, gradient descent! Right?"


def test_highlight(vocab):
    text = "Use gradient descent on the CPU."
    assert highlight_terms(text, path=vocab) == "Use **gradient descent** on the **CPU**."
    assert [s["term"] for s in find_terms(text, vocab)] == ["gradient descent", "CPU"]


def test_initial_prompt(vocab):
    prompt = build_initial_prompt(vocab)
    assert "Dr. Ramirez" in prompt and "gradient descent" in prompt and "TCP" in prompt
