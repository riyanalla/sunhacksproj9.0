"""Split live mic audio into utterances at natural pauses, for low-latency captions.

Instead of waiting for a fixed 4 s chunk, an utterance is sent to Whisper as soon as the
speaker pauses. Silence is never sent at all. If someone talks for a long time without
pausing, the audio is cut at the most recent short gap (between words) so words aren't
split in half. Quiet mics are boosted to a normal level first (AutoGain), so a laptop
across the room doesn't lose words to the speech detector.

    chunker = PauseChunker()
    for start_sample, audio in chunker.feed(samples):   # call with each mic block
        transcribe(audio)
    for start_sample, audio in chunker.flush():         # on stop: whatever is left
        transcribe(audio)
"""

from __future__ import annotations

import numpy as np
import webrtcvad

SAMPLE_RATE = 16000
FRAME = 480                 # 30 ms, one of the sizes webrtcvad accepts
FRAME_S = FRAME / SAMPLE_RATE
CUT_SEARCH_FRAMES = 50      # forced cuts look back this far (1.5 s) for the quietest spot


class AutoGain:
    """Boost quiet mics (e.g. a laptop at the back of the room) to a normal speech level.

    Only frames clearly louder than the background (3x the noise floor) count as speech;
    the speech level is the 90th percentile of the last 100 of those (~3 s of talking).
    So the gain settles within a few seconds, and holds steady through pauses instead of
    pumping up room noise. Only ever boosts (gain >= 1): a loud enough mic is untouched.
    """

    SPEECH_FRAMES = 100  # ~3 s of speech
    NOISE_FRAMES = 300   # ~9 s of history for the background level

    def __init__(self, target_rms: float = 0.02, max_gain: float = 20.0):
        self.target, self.max_gain = target_rms, max_gain
        self._all: list[float] = []
        self._speech: list[float] = []
        self.gain = 1.0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self._all = self._all[-(self.NOISE_FRAMES - 1):] + [rms]
        noise = float(np.percentile(self._all, 10))
        if len(self._all) >= 10 and rms > 3 * noise:
            self._speech = self._speech[-(self.SPEECH_FRAMES - 1):] + [rms]
            if len(self._speech) >= 10:
                level = float(np.percentile(self._speech, 90))
                self.gain = min(self.max_gain, max(1.0, self.target / max(level, 1e-6)))
        return np.clip(frame * self.gain, -1.0, 1.0)


class PauseChunker:
    def __init__(self, pause: float = 0.5, max_seconds: float = 6.0,
                 min_speech: float = 0.25, preroll: float = 0.3, aggressiveness: int = 2,
                 auto_gain: bool = True):
        self.gain = AutoGain() if auto_gain else None
        self.vad = webrtcvad.Vad(aggressiveness)
        self.end_frames = max(1, round(pause / FRAME_S))         # silence that ends an utterance
        self.max_frames = round(max_seconds / FRAME_S)           # force a cut after this long
        self.min_speech_frames = round(min_speech / FRAME_S)     # shorter blips are noise
        self.preroll_frames = round(preroll / FRAME_S)           # keep a bit before speech starts
        self._leftover = np.zeros(0, dtype=np.float32)
        self._pos = 0               # absolute sample index of the next whole frame
        self._frames: list[np.ndarray] = []   # current utterance (or preroll when idle)
        self._flags: list[bool] = []          # speech/non-speech per frame in _frames
        self._start = 0             # absolute sample index of _frames[0]
        self._active = False
        self._silence = 0

    def feed(self, samples: np.ndarray) -> list[tuple[int, np.ndarray]]:
        out = []
        buf = np.concatenate([self._leftover, np.asarray(samples, dtype=np.float32)])
        n = len(buf) // FRAME * FRAME
        self._leftover = buf[n:]
        for i in range(0, n, FRAME):
            out += self._frame(buf[i:i + FRAME])
        return out

    def flush(self) -> list[tuple[int, np.ndarray]]:
        out = self._emit(len(self._frames)) if self._active else []
        self._frames, self._flags, self._active, self._silence = [], [], False, 0
        self._start = self._pos
        return out

    def _frame(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        if self.gain:
            frame = self.gain(frame)  # quiet mic: boost before the VAD and Whisper hear it
        pcm = (np.clip(frame, -1, 1) * 32767).astype(np.int16).tobytes()
        speech = self.vad.is_speech(pcm, SAMPLE_RATE)
        self._frames.append(frame)
        self._flags.append(speech)
        self._pos += FRAME

        if not self._active:
            if speech:
                self._active, self._silence = True, 0
            elif len(self._frames) > self.preroll_frames:  # idle: keep only the preroll
                self._frames.pop(0)
                self._flags.pop(0)
                self._start += FRAME
            return []

        self._silence = 0 if speech else self._silence + 1
        if self._silence >= self.end_frames:  # speaker paused: send it now
            # Keep ~half the pause as tail so the last word isn't clipped.
            keep = len(self._frames) - self._silence + self.end_frames // 2
            out = self._emit(keep)
            self._frames, self._flags, self._active = [], [], False
            self._start = self._pos
            return out
        if len(self._frames) >= self.max_frames:  # long stretch: cut between words
            return self._emit(self._cut_point())
        return []

    def _cut_point(self) -> int:
        """Where to split a long stretch of speech so no word is cut in half.

        First choice: the middle of the latest VAD gap (>= 2 quiet frames) in the second
        half of the buffer. On noisy mics the VAD can call everything speech, so the
        fallback is the quietest moment of the last ~1.5 s, which is almost always
        between words.
        """
        n = len(self._flags)
        i = n - 1
        while i > n // 2:
            if not self._flags[i] and not self._flags[i - 1]:
                j = i
                while j > n // 2 and not self._flags[j - 1]:
                    j -= 1
                return (i + j + 1) // 2
            i -= 1
        lo = max(n // 2, n - CUT_SEARCH_FRAMES)
        energy = np.array([float(np.mean(f ** 2)) for f in self._frames[lo:n]])
        if len(energy) < 3:
            return n
        smooth = np.convolve(energy, np.ones(3) / 3, mode="same")  # ignore one-frame dips
        return lo + int(np.argmin(smooth)) + 1

    def _emit(self, k: int) -> list[tuple[int, np.ndarray]]:
        """Send the first k frames as an utterance; keep the rest buffered."""
        k = max(1, min(k, len(self._frames)))
        head, flags = self._frames[:k], self._flags[:k]
        start = self._start
        self._frames, self._flags = self._frames[k:], self._flags[k:]
        self._start += k * FRAME
        if sum(flags) < self.min_speech_frames:
            return []
        return [(start, np.concatenate(head))]
