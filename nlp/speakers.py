"""Speaker labels: voice fingerprints (resemblyzer) plus a pause-based fallback."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000  # faster-whisper works on 16 kHz mono float32
UNKNOWN = "Unknown"


def _prep(audio, preprocess_wav):
    """Accept a file path or a 16 kHz float32 numpy array."""
    if isinstance(audio, (str, Path)):
        return preprocess_wav(Path(audio))
    return preprocess_wav(np.asarray(audio, dtype=np.float32), source_sr=SAMPLE_RATE)


class SpeakerIdentifier:
    """Labels each segment with the closest enrolled voice, or "Unknown"."""

    def __init__(self, threshold: float = 0.75, min_seconds: float = 0.8):
        from resemblyzer import VoiceEncoder, preprocess_wav  # heavy import, load once
        self._preprocess = preprocess_wav
        self.encoder = VoiceEncoder()  # add device="cuda" if you have a GPU
        self.threshold = threshold        # min cosine similarity to accept a match
        self.min_seconds = min_seconds    # shorter segments are too noisy to judge
        self.fingerprints: dict[str, np.ndarray] = {}

    def enroll(self, name: str, audio) -> None:
        """Add one clip of `name` talking alone. Several clips per person get averaged."""
        self.enroll_many(name, [audio])

    def enroll_many(self, name: str, clips) -> None:
        wavs = [_prep(c, self._preprocess) for c in clips]
        embs = [self.encoder.embed_utterance(w) for w in wavs if len(w)]
        if not embs:
            raise ValueError(f"No speech found in enrollment audio for {name!r}")
        mean = np.mean(embs, axis=0)
        self.fingerprints[name] = mean / np.linalg.norm(mean)

    def enroll_dir(self, folder: str | Path) -> list[str]:
        """Enroll from files like enroll/alice.wav, enroll/alice_2.wav, enroll/bob.wav."""
        groups: dict[str, list[Path]] = defaultdict(list)
        for f in sorted(Path(folder).glob("*.wav")):
            groups[f.stem.split("_")[0]].append(f)
        for name, files in groups.items():
            self.enroll_many(name, files)
        return list(groups)

    def identify(self, audio) -> tuple[str | None, float]:
        """Return (speaker, similarity). Speaker is None if the clip is too short to judge."""
        if not self.fingerprints:
            return UNKNOWN, 0.0
        wav = _prep(audio, self._preprocess)
        if len(wav) < self.min_seconds * SAMPLE_RATE:
            return None, 0.0
        emb = self.encoder.embed_utterance(wav)
        scores = {name: float(np.dot(emb, fp)) for name, fp in self.fingerprints.items()}
        best = max(scores, key=scores.get)
        if scores[best] < self.threshold:
            return UNKNOWN, scores[best]
        return best, scores[best]


class PauseSpeakerLabeler:
    """9 PM fallback: assume a new speaker after a pause longer than `gap` seconds."""

    def __init__(self, gap: float = 1.5, names: list[str] | None = None):
        self.gap = gap
        self.names = names or ["Speaker 1", "Speaker 2"]
        self._idx = 0
        self._last_end: float | None = None

    def label(self, start: float, end: float) -> str:
        if self._last_end is not None and start - self._last_end > self.gap:
            self._idx = (self._idx + 1) % len(self.names)
        self._last_end = end
        return self.names[self._idx]
