"""Whisper settings shared by live captions and the accuracy test, so they always match.

    tr = Transcriber("tiny.en")
    for seg in tr(chunk):              # one utterance from PauseChunker
        print(seg.start, seg.text)

    segments = transcribe_audio(tr, audio)   # whole recording, chunked like live mode
"""

from __future__ import annotations

from .chunker import PauseChunker
from .vocab import build_hotwords

SAMPLE_RATE = 16000
CONTEXT_CHARS = 200  # how much of the previous transcript to carry into the next chunk
FEED_BLOCK = 1600    # 0.1 s, roughly what the mic callback delivers


class Transcriber:
    """faster-whisper with course hotwords and the previous sentence as context."""

    def __init__(self, model_size: str = "base", beam_size: int = 1):
        from faster_whisper import WhisperModel
        self.model_size = model_size
        self.beam_size = beam_size
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.hotwords = build_hotwords()  # course + classroom words, sent with every chunk
        self.recent = ""  # tail of what was just said, so each chunk hears the one before

    def __call__(self, chunk) -> list:
        segments, _ = self.model.transcribe(
            chunk, hotwords=self.hotwords, initial_prompt=self.recent or None,
            language="en", vad_filter=True, beam_size=self.beam_size)
        out = []
        for seg in segments:  # generator: decoding happens as we iterate
            # Raw Whisper text (before term fixes) is the most natural context for it.
            self.recent = (self.recent + " " + seg.text.strip())[-CONTEXT_CHARS:]
            out.append(seg)
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
