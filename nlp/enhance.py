"""Person 2's hook for the ASR pipeline: add the speaker and fix course terms.

    from nlp.enhance import setup, enhance

    setup()                                    # once at startup (pause-based speakers)
    caption = {"type": "caption", "session_id": "lecture-1", "segment_id": 42,
               "speaker": None, "text": "we use gibs sampling", "start": 12.4,
               "end": 15.1, "is_final": True}
    caption = enhance(caption, audio=segment_audio)
    # -> speaker "Speaker 1", text "we use Gibbs sampling"; then POST it to the server

Input and output follow PROTOCOL.md. `enhance` returns a new dict (the input is not
modified), never raises, and leaves every field except "speaker" and "text" untouched.

- Final captions (is_final true or missing): the speaker is decided and remembered.
- Partials (is_final false): text is fixed too, and the speaker is the best guess for
  this segment, without changing speaker history, so drafts never shift the labels.
- `audio` (16 kHz mono float32 numpy, just this segment) is only needed for voice-based
  speakers (setup(mode="embed")). Pause-based speakers use "start"/"end".
- A "session_start" message resets speaker history; other types pass through unchanged.
- Slides: once the server calls `nlp.context.set_context(session_id, slides_text)`, key
  terms from the slides are corrected too, for captions with that "session_id".
"""

from __future__ import annotations

import traceback

from . import context, label
from .vocab import fix_terms

_ready = False


def setup(mode: str = "pause") -> None:
    """mode="pause": new speaker after a long pause. mode="embed": by voice (needs enroll/)."""
    global _ready
    label.setup(mode=mode)
    _ready = True


def enhance(caption: dict, audio=None) -> dict:
    """Return a copy of a PROTOCOL.md caption with "speaker" set and "text" fixed."""
    out = dict(caption)
    try:
        if not _ready:
            setup()
        kind = out.get("type", "caption")
        if kind == "session_start":
            label.reset()
            return out
        if kind != "caption":
            return out

        text = out.get("text") or ""
        start, end = out.get("start"), out.get("end")
        terms = context.fix_terms_for(out.get("session_id"))  # from uploaded slides
        if out.get("is_final", True):
            speaker, fixed = label.label_segment(audio, text, start, end, terms)
        else:
            speaker, fixed = label.current_speaker(start), fix_terms(text, extra_terms=terms)
        out["speaker"], out["text"] = speaker, fixed
    except Exception:
        traceback.print_exc()  # never break the caption stream
    return out
