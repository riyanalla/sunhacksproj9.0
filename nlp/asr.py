"""Whisper settings shared by live captions and the accuracy test, so they always match.

    tr = Transcriber("tiny.en")
    for seg in tr(chunk):              # one utterance from PauseChunker
        print(seg.start, seg.text)

    segments = transcribe_audio(tr, audio)   # whole recording, chunked like live mode
"""

from __future__ import annotations

import dataclasses
import re

from . import context
from .chunker import PauseChunker
from .vocab import build_hotwords

SAMPLE_RATE = 16000
CONTEXT_CHARS = 200  # how much of the previous transcript to carry into the next chunk
FEED_BLOCK = 1600    # 0.1 s, roughly what the mic callback delivers

# Whisper's usual inventions on noise/silence (it learned them from YouTube subtitles).
HALLUCINATIONS = re.compile(
    r"thanks? (you )?for watching|subscribe|like and subscribe|see you (in the )?next|"
    r"subtitles? by|transcribed by|captions? by|amara\.org|thank you\.?$", re.I)
# Short words that can legitimately start a chunk in lowercase. Anything else of 1-2
# letters at the very start is the tail of a word cut in half ("nd", "ng", "ed").
SHORT_WORDS = {"a", "i", "an", "am", "as", "at", "be", "by", "do", "go", "he", "if", "in",
               "is", "it", "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to", "up",
               "us", "we"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def clean_text(text: str, previous: str = "") -> str:
    """Remove chunk-boundary junk from one Whisper segment.

    - a word fragment at the start ("nd to view this" -> "to view this"), left over when
      a long stretch of speech was cut mid-word
    - words repeated across the boundary: previous ended "...frequentist interpretation",
      this one starts "interpretation, which" -> "which"
    """
    text = text.strip()
    m = re.match(r"([a-z]{1,2})\b[,.]?\s*", text)  # lowercase only: real sentences start capitalized
    if m and m.group(1) not in SHORT_WORDS:
        text = text[m.end():]

    prev, cur = _words(previous), _words(text)
    for k in range(min(len(prev), len(cur), 6), 0, -1):
        if prev[-k:] == cur[:k] and (k >= 2 or len(cur[0]) >= 5):
            # drop the first k words (with their punctuation) from the original text
            m = re.match(r"(\W*[\w']+){%d}[\W]*" % k, text)
            text = text[m.end():] if m else text
            break
    return text


def is_junk(seg) -> bool:
    """Segments that are almost certainly not speech: Whisper's own scores say so."""
    if seg.no_speech_prob > 0.6 and seg.avg_logprob < -1.0:  # Whisper's standard silence rule
        return True
    if seg.compression_ratio > 2.4:  # "the the the the..." repetition loops
        return True
    if seg.avg_logprob < -1.2:       # very unsure: on a far mic this is noise, not words
        return True
    return bool(HALLUCINATIONS.search(seg.text)) and seg.avg_logprob < -0.5


class Transcriber:
    """faster-whisper with course hotwords and the previous sentence as context.

    Returned segments are cleaned: noise and hallucinations are dropped, and word
    fragments / repeated words at chunk boundaries are removed (see clean_text, is_junk).

    With a session_id, the key terms of that lecture's uploaded slides (context.py) are
    sent to Whisper too, and picked up as soon as they're uploaded.
    """

    def __init__(self, model_size: str = "base", beam_size: int = 1, session_id=None):
        from faster_whisper import WhisperModel
        self.model_size = model_size
        self.beam_size = beam_size
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.session_id = session_id
        self._hotwords_for = None  # (session_id, context version) the cache was built for
        self._hotwords = ""
        self.recent = ""  # tail of what was just said, so each chunk hears the one before

    @property
    def hotwords(self) -> str:
        """Course + slide + classroom words, sent with every chunk."""
        key = (self.session_id, context.version(self.session_id))
        if key != self._hotwords_for:
            self._hotwords = build_hotwords(extra=context.hint_terms(self.session_id))
            self._hotwords_for = key
        return self._hotwords

    def set_session(self, session_id) -> None:
        """Switch to another lecture: its slide terms, and no context from the last one."""
        self.session_id, self.recent = session_id, ""

    def __call__(self, chunk) -> list:
        segments, _ = self.model.transcribe(
            chunk, hotwords=self.hotwords, initial_prompt=self.recent or None,
            language="en", vad_filter=True, beam_size=self.beam_size)
        out = []
        for seg in segments:  # generator: decoding happens as we iterate
            if is_junk(seg):
                continue
            text = clean_text(seg.text, self.recent)
            if not _words(text):
                continue
            # Whisper's text (before term fixes) is the most natural context for it.
            self.recent = (self.recent + " " + text)[-CONTEXT_CHARS:]
            out.append(dataclasses.replace(seg, text=" " + text))
        return out


def transcribe_audio(tr: Transcriber, audio, pause: float = 0.5,
                     max_seconds: float = 6.0) -> list[tuple[float, float, str]]:
    """Run a whole recording through the same chunking as live mode.

    Returns (start_s, end_s, raw_text) per Whisper segment, in recording time.
    """
    chunker = PauseChunker(pause=pause, max_seconds=max_seconds)
    utterances = []
    for i in range(0, len(audio), FEED_BLOCK):
        utterances += chunker.feed(audio[i:i + FEED_BLOCK])
    utterances += chunker.flush()

    out = []
    for start, chunk in utterances:
        offset = start / SAMPLE_RATE
        for seg in tr(chunk):
            out.append((offset + seg.start, offset + seg.end, seg.text.strip()))
    return out
