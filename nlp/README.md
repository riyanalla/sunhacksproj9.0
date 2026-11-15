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

## Handoff to Person 1 (ASR): `enhance()`

Team plan: Pi → server `/ws/audio` → server forwards frames → ASR (Person 1) →
`enhance()` (this folder) → `POST /sessions/{id}/caption` → phones + database.

Call `enhance` on every caption right before POSTing it. It takes and returns a caption
in the `PROTOCOL.md` format, sets `"speaker"` and fixes course terms in `"text"`:

```python
from nlp.enhance import setup, enhance

setup()                                     # once at startup (pause-based speakers)

caption = {"type": "caption", "session_id": "lecture-1", "segment_id": 42,
           "speaker": None, "text": "we use gibs sampling", "start": 12.4,
           "end": 15.1, "is_final": True}
caption = enhance(caption, audio=segment_audio)   # audio is optional, see below
requests.post(f"{SERVER}/sessions/lecture-1/caption", json=caption)
```

- **Never raises** and never modifies the dict you pass in. Fields other than `speaker`
  and `text` are left as they are.
- **Partials** (`is_final: false`): call it on drafts too. The text gets fixed and the
  speaker is the best guess for that segment, but drafts don't change speaker history.
- **`audio`**: this segment's audio as 16 kHz mono float32 numpy. Only needed for
  voice-based speakers (`setup(mode="embed")`); pause-based speakers use `start`/`end`,
  which must be in seconds since the session started.
- **`session_start`** messages reset the speakers; pass them through `enhance` too.

### Optional: reuse this folder's ASR pieces

`asr.Transcriber` and `chunker.PauseChunker` are the tuned Whisper settings and chunking
behind the accuracy numbers above. To use them on frames from the server:

```python
import numpy as np
from nlp.asr import Transcriber
from nlp.chunker import PauseChunker

whisper, chunker = Transcriber("base"), PauseChunker()

def on_frame(pcm_bytes):                     # 100 ms of 16 kHz int16 from the Pi
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768
    for start_sample, chunk in chunker.feed(audio):
        offset = start_sample / 16000
        for seg in whisper(chunk):
            seg_audio = chunk[int(seg.start * 16000):int(seg.end * 16000)]
            # build the caption with start=offset + seg.start, end=offset + seg.end,
            # then enhance(caption, audio=seg_audio) and POST it
```

Run Whisper off the thread that receives frames, so a slow chunk never blocks the audio,
and call `chunker.flush()` when the session ends to caption the last few seconds.

Translation (`Translator("es")(text)`, never raises) isn't in `PROTOCOL.md` yet; agree on
a field with the team before sending it.

## Files

| File | What it does |
|---|---|
| `enhance.py` | `enhance(caption, audio)`: the hook Person 1 calls before POSTing each caption |
| `vocab.json` | Course terms, acronyms, professor name, classroom words. **Edit this per course.** |
| `vocab.py` | `build_hotwords()`, `fix_terms(text)`, `highlight_terms(text)`, `find_terms(text)` |
| `asr.py` | `Transcriber`: the Whisper settings shared by live captions and the accuracy test |
| `chunker.py` | `PauseChunker`: splits live audio into utterances at pauses |
| `translate.py` | `Translator`: offline translation; `python -m nlp.translate --list` |
| `label.py`, `speakers.py` | Speaker labels: pause-based, or voice fingerprints (resemblyzer) |
| `live_test.py` | Live mic captions |
| `wer.py`, `wer_test.py` | Word error rate scoring and the accuracy test |
| `scripts/` | Texts to read for the accuracy test |
| `test_*.py` | `python -m pytest nlp` |

## Tuning

- `live_test --chunk 4` shows captions sooner but makes more errors; `--pause 0.4` ends
  utterances on shorter pauses.
- `vocab.SCORE_CUTOFF = 85`, `COMMON_WORDS`: add a word to `COMMON_WORDS` if it keeps getting
  "fixed" by mistake.
- `SpeakerIdentifier(threshold=0.75)`: raise it if voices get mixed up, lower it if too many
  segments come back as "Unknown".
