"""Handoff to Person 1's pipeline.

    from nlp.label import setup, label_segment

    setup()                                   # once at startup (loads the voice model)
    for seg in segments:                      # after Whisper, before sending to the server
        audio = full_audio[int(seg.start * 16000):int(seg.end * 16000)]
        speaker, text = label_segment(audio, seg.text, seg.start, seg.end)

`audio` is 16 kHz mono float32 numpy (what faster_whisper.decode_audio returns).
`start`/`end` (seconds) are only needed for the pause fallback.

Set NLP_SPEAKER_MODE=pause (or setup(mode="pause")) to switch to the fallback.
setup(mode="single") labels everything SINGLE_SPEAKER ("Professor"): for a lecture, where
the pause fallback would flip speakers every time the professor pauses.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from .speakers import UNKNOWN, PauseSpeakerLabeler, SpeakerIdentifier
from .vocab import fix_terms

ENROLL_DIR = Path(__file__).with_name("enroll")
SINGLE_SPEAKER = "Professor"

_mode = os.environ.get("NLP_SPEAKER_MODE", "embed")
_identifier: SpeakerIdentifier | None = None
_pause = PauseSpeakerLabeler()
_last_speaker = UNKNOWN


def setup(mode: str | None = None, enroll_dir: str | Path = ENROLL_DIR) -> None:
    """Load the voice encoder and enroll teammates. Falls back to pause mode on failure."""
    global _mode, _identifier
    if mode:
        _mode = mode
    if _mode != "embed":  # "pause" and "single" need no model
        return
    try:
        _identifier = SpeakerIdentifier()
        names = _identifier.enroll_dir(enroll_dir)
        print(f"[nlp] enrolled speakers: {names}", file=sys.stderr)
        if not names:
            raise RuntimeError(f"no .wav files in {enroll_dir}")
    except Exception:
        traceback.print_exc()
        print("[nlp] speaker ID unavailable, using pause fallback", file=sys.stderr)
        _mode = "pause"


def reset() -> None:
    """Forget speaker history, e.g. at the start of a new lecture. Keeps enrolled voices."""
    global _pause, _last_speaker
    _pause = PauseSpeakerLabeler()
    _last_speaker = UNKNOWN


def current_speaker(start: float | None = None) -> str:
    """Best guess for an in-progress (partial) segment, without changing any state."""
    try:
        if _mode == "single":
            return SINGLE_SPEAKER
        if _mode == "pause" and start is not None:
            return _pause.peek(start)
    except Exception:
        traceback.print_exc()
    return _last_speaker


def label_segment(audio, text: str, start: float | None = None,
                  end: float | None = None, terms: tuple[str, ...] = ()) -> tuple[str, str]:
    """(speaker, fixed_text) for one Whisper segment. Never raises.

    `terms`: extra terms to correct toward, on top of vocab.json (this lecture's slides).
    """
    global _last_speaker
    try:
        fixed = fix_terms(text, extra_terms=terms)
    except Exception:
        traceback.print_exc()
        fixed = text

    try:
        if _mode == "single":
            _last_speaker = SINGLE_SPEAKER
        elif _mode == "pause":
            if start is None or end is None:
                return _last_speaker, fixed
            _last_speaker = _pause.label(start, end)
        else:
            if _identifier is None:
                setup()
                if _mode == "pause":
                    return label_segment(audio, text, start, end, terms)
            speaker, _score = _identifier.identify(audio)
            if speaker is not None:  # None = too short to tell; keep previous label
                _last_speaker = speaker
    except Exception:
        traceback.print_exc()
    return _last_speaker, fixed
