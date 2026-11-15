"""Live audio -> finished captions, for the server's /ws/audio endpoint.

    from nlp.stream import LiveStream

    stream = LiveStream("6551f0c2...")            # one per audio connection
    for caption in stream.feed_pcm(frame_bytes):  # 16 kHz mono int16 (PROTOCOL.md)
        save_and_broadcast(caption)
    for caption in stream.flush():                # when the mic stops
        save_and_broadcast(caption)

Each caption is a PROTOCOL.md "caption" message (is_final true) that has already been
through `enhance()`: speaker set, course and slide terms fixed. Uses everything tuned in
this folder: AutoGain + PauseChunker (chunker.py), Whisper with hints and cleanup
(asr.py), slide terms for the session (context.py).

`feed_pcm` blocks while Whisper runs (about 1-3 s per utterance on a laptop CPU), so call
it from a worker thread, never from the server's event loop.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import enhance
from .asr import SAMPLE_RATE, Transcriber
from .chunker import PauseChunker

_models: dict[str, object] = {}
_models_lock = threading.Lock()
_whisper_lock = threading.Lock()  # one Whisper call at a time: the CPU is the bottleneck


def shared_model(model_size: str = "base"):
    """Load a Whisper model once per process and share it between streams."""
    with _models_lock:
        if model_size not in _models:
            from faster_whisper import WhisperModel
            _models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        return _models[model_size]


class LiveStream:
    def __init__(self, session_id: str, model_size: str = "base", start_offset: float = 0.0,
                 first_segment_id: int | None = None, pause: float = 0.5,
                 max_seconds: float = 6.0):
        """
        start_offset: seconds into the lecture when this audio starts (if the mic
            reconnects, captions keep counting from where the lecture is).
        first_segment_id: segment ids count up from here; defaults to the current time
            in ms, so a reconnect never reuses (and overwrites) an earlier segment id.
        """
        self.session_id = str(session_id)
        self.offset = float(start_offset)
        self.next_id = int(time.time() * 1000) if first_segment_id is None else first_segment_id
        self.chunker = PauseChunker(pause=pause, max_seconds=max_seconds)
        self.tr = Transcriber(model_size, session_id=self.session_id,
                              model=shared_model(model_size))
        self._odd = b""  # a frame can end mid-sample; keep the stray byte

    def feed_pcm(self, pcm: bytes) -> list[dict]:
        """Raw 16 kHz mono int16 little-endian audio -> captions finished by it (often [])."""
        data = self._odd + bytes(pcm)
        cut = len(data) - len(data) % 2
        self._odd = data[cut:]
        audio = np.frombuffer(data[:cut], dtype="<i2").astype(np.float32) / 32768.0
        return self._captions(self.chunker.feed(audio))

    def feed_audio(self, audio: np.ndarray) -> list[dict]:
        """Same as feed_pcm, for float32 audio in [-1, 1] (e.g. a test recording)."""
        return self._captions(self.chunker.feed(np.asarray(audio, dtype=np.float32)))

    def flush(self) -> list[dict]:
        """Caption whatever is still buffered (call when the mic stops)."""
        return self._captions(self.chunker.flush())

    def _captions(self, utterances) -> list[dict]:
        out = []
        for start_sample, chunk in utterances:
            offset = self.offset + start_sample / SAMPLE_RATE
            with _whisper_lock:
                segments = self.tr(chunk)
            for seg in segments:
                caption = {
                    "type": "caption", "session_id": self.session_id,
                    "segment_id": self.next_id, "speaker": None,
                    "text": seg.text.strip(),
                    "start": round(offset + seg.start, 2), "end": round(offset + seg.end, 2),
                    "is_final": True,
                }
                seg_audio = chunk[int(seg.start * SAMPLE_RATE):int(seg.end * SAMPLE_RATE)]
                caption = enhance.enhance(caption, audio=seg_audio)
                if caption.get("text"):
                    self.next_id += 1
                    out.append(caption)
        return out
