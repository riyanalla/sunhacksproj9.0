# nlp: live captions, course vocabulary, translation, speaker labels

Turns lecture audio into captions: transcribes with Whisper (faster-whisper, CPU), fixes
course terms, labels the speaker, and can translate each caption into another language.

## Setup (Windows)

Run everything from the repo root.

```
python -m venv nlp\.venv
nlp\.venv\Scripts\activate
pip install -r nlp\requirements.txt
pip install resemblyzer --no-deps
python -m nlp.download_model base          # cache Whisper while online (also: tiny.en)
python -m nlp.translate es                 # cache any translation languages you want
```

## Live captions

```
python -m nlp.live_test                    # live captions from the mic
python -m nlp.live_test --translate es     # with Spanish preloaded
python -m nlp.live_test --model tiny.en    # faster, less accurate
python -m nlp.live_test --list-devices     # find your mic, then add --device N
```

Once it says `Ready`, type a command and press Enter:

```
start            begin captions (plain Enter also toggles start/stop)
start spanish    captions + Spanish translation (french, hindi, es, ...; works mid-recording)
start english    turn translation off
stop             pause; whatever was said so far still gets transcribed
quit             exit
```

`python -m nlp.translate --list` shows every language available from English.

## Accuracy test (word error rate)

```
python -m nlp.wer_test --ref nlp/scripts/course_description.txt   # read it aloud, get WER
python -m nlp.wer_test                                            # or paste any text
python -m nlp.wer_test --audio nlp/wer_runs/run_0001.wav --model tiny.en   # re-score, other settings
python -m nlp.wer_test --history                                  # compare every run
```

Captions show live while you read, then you get the WER and a list of every mistake.
Recordings and results are saved in `nlp/wer_runs/` (git-ignored).

On a 4-core laptop CPU, reading the course description: `base` with 6 s chunks gets
about 3-7% WER at 4-5x real time; `tiny.en` is faster but makes roughly twice the errors.

## How it works

1. **Chunking** (`chunker.py`): audio goes to Whisper when the speaker pauses (0.5 s), or
   after at most 6 s, cut at the quietest moment so words aren't split. Silence is skipped.
2. **Transcription** (`asr.py`): Whisper gets the course vocabulary as `hotwords` and the
   previous sentence as `initial_prompt`, so each chunk has context.
3. **Term fixing** (`vocab.py`): near-miss spellings ("basian", "gibs sampling") are
   corrected to the spelling in `vocab.json`.
4. **Speaker labels** (`label.py`): pause-based guess by default; voice-based after
   `python -m nlp.live_test enroll <name>` for each speaker (`--mode embed`).
5. **Translation** (`translate.py`): offline English -> other language (Argos Translate).

## Handoff to the server

The Pi sends 16 kHz mono int16 PCM (see `PROTOCOL.md`). Convert each frame to float32
and run it through the same pieces `live_test` uses:

```python
import numpy as np
from nlp.asr import Transcriber
from nlp.chunker import PauseChunker
from nlp.label import setup, label_segment
from nlp.translate import Translator      # optional

setup(mode="pause")                        # once at startup
whisper, chunker = Transcriber("base"), PauseChunker()

def on_pi_frame(pcm_bytes):
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768
    for start_sample, chunk in chunker.feed(audio):
        offset = start_sample / 16000
        for seg in whisper(chunk):
            seg_audio = chunk[int(seg.start * 16000):int(seg.end * 16000)]
            speaker, text = label_segment(seg_audio, seg.text.strip(),
                                          offset + seg.start, offset + seg.end)
            # send {"speaker": speaker, "text": text, "start": offset + seg.start,
            #       "end": offset + seg.end, "is_final": True} to the phones
```

- `label_segment` never raises. If something fails it returns the previous speaker and the
  unfixed text.
- Call `chunker.flush()` when a session ends to caption the last few seconds.
- `Translator("es")(text)` returns the Spanish text; it also never raises.

## Files

| File | What it does |
|---|---|
| `vocab.json` | Course terms, acronyms, professor name, classroom words. **Edit this per course.** |
| `vocab.py` | `build_hotwords()`, `fix_terms(text)`, `highlight_terms(text)`, `find_terms(text)` |
| `asr.py` | `Transcriber`: the Whisper settings shared by live captions and the accuracy test |
| `chunker.py` | `PauseChunker`: splits live audio into utterances at pauses |
| `translate.py` | `Translator`: offline translation; `python -m nlp.translate --list` |
| `label.py`, `speakers.py` | Speaker labels: pause-based, or voice fingerprints (resemblyzer) |
| `live_test.py` | Live mic captions |
| `wer.py`, `wer_test.py` | Word error rate scoring and the accuracy test |
| `scripts/` | Texts to read for the accuracy test |
| `test_vocab.py`, `test_wer.py` | `python -m pytest nlp` |

## Tuning

- `live_test --chunk 4` shows captions sooner but makes more errors; `--pause 0.4` ends
  utterances on shorter pauses.
- `vocab.SCORE_CUTOFF = 85`, `COMMON_WORDS`: add a word to `COMMON_WORDS` if it keeps getting
  "fixed" by mistake.
- `SpeakerIdentifier(threshold=0.75)`: raise it if voices get mixed up, lower it if too many
  segments come back as "Unknown".
