"""python -m pytest nlp/test_wer.py"""

from nlp.wer import normalize, score


def test_formatting_is_not_an_error():
    assert score("OK, submit it by Friday!", "okay submit it by friday").wer == 0
    assert score("Late work loses 10% per day.", "late work loses ten percent per day").wer == 0
    assert score("Read chapter 4th.", "read chapter fourth").wer == 0


def test_numbers_normalize():
    assert normalize("1,250") == "one thousand two hundred fifty".split()
    assert normalize("2.5") == "two point five".split()
    assert normalize("21st 30th") == "twenty first thirtieth".split()


def test_substitution():
    r = score("submit your assignments by Friday", "submit your airplanes by Friday")
    assert r.wer == 1 / 5
    assert (r.errors[0].said, r.errors[0].heard) == ("assignments", "airplanes")
    assert "[assignments -> airplanes]" in r.diff


def test_missed_and_added_words():
    r = score("the homework is due today", "the homework due today today")
    assert (r.deletions, r.insertions, r.substitutions) == (1, 1, 0)
    assert r.wer == 2 / 5


def test_neighbouring_errors_group():
    r = score("lose ten percent per day", "lose tenth 1.5 percent per day")
    assert len(r.errors) == 1
    assert r.errors[0].said == "ten"
