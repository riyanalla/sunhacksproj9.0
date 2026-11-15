"""Live mic test: talk and see speaker labels + fixed terms printed as you go.

    python -m nlp.live_test                        # live captions (speaker = pause-based guess)
    python -m nlp.live_test --translate es         # start with Spanish already picked
    python -m nlp.live_test --model tiny.en        # faster, less accurate
    python -m nlp.live_test --list-devices         # find your mic, then add --device N
    python -m nlp.live_test enroll alice           # record 10 s of alice talking alone,
    python -m nlp.live_test --mode embed           # then label speakers by voice

While it runs, type a command and press Enter:
    start           begin recording (keeps the last language; plain Enter toggles start/stop)
    start spanish   record and translate captions into Spanish (also: french, hindi, es, ...)
    start english   record with English captions only (turns translation off)
    stop            pause recording; whatever was said so far gets transcribed
    quit            exit
Typing "start french" while already recording switches the language on the fly.

Audio is sent to Whisper each time the speaker pauses (--pause, default 0.5 s), or
after at most --chunk seconds (default 6), cut at a gap between words. Silence is
never transcribed. See chunker.py.
"""

import argparse
import queue
import sys
import threading
import time

import sounddevice as sd
import soundfile as sf

from .chunker import PauseChunker
from .speakers import SAMPLE_RATE

ENROLL_SECONDS = 10


def record_enrollment(name: str, device=None):
    from .label import ENROLL_DIR
    ENROLL_DIR.mkdir(exist_ok=True)
    path = ENROLL_DIR / f"{name}.wav"
    if path.exists():  # keep earlier clips; they get averaged together
        i = 2
        while (ENROLL_DIR / f"{name}_{i}.wav").exists():
            i += 1
        path = ENROLL_DIR / f"{name}_{i}.wav"
    input(f"{name}: press Enter, then talk normally for {ENROLL_SECONDS} s...")
    audio = sd.rec(ENROLL_SECONDS * SAMPLE_RATE, samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32", device=device)
    for s in range(ENROLL_SECONDS, 0, -1):
        print(f"  recording... {s}", end="\r")
        time.sleep(1)
    sd.wait()
    sf.write(path, audio[:, 0], SAMPLE_RATE)
    print(f"\nSaved {path}")


def live(mode: str, chunk_seconds: float, model_size: str, device=None,
         translate_to=None, pause_seconds: float = 0.5):
    from . import label
    from .asr import Transcriber
    from .vocab import highlight_terms
    from .translate import SOURCE, Translator, language_name, resolve_language

    translators: dict = {}  # code -> loaded Translator, so switching back is instant
    current = {"code": None, "tr": None}  # language captions are translated into

    def set_language(word) -> bool:
        """Point captions at `word` (name or code). Loads/downloads the model if needed."""
        code = resolve_language(word) if word else None
        if code in (None, SOURCE):
            current.update(code=None, tr=None)
            print("Captions: English only")
            return True
        if code not in translators:
            print(f"Loading {language_name(code)} translator...")
            try:
                translators[code] = Translator(code)
            except Exception as e:
                print(f"Can't translate to {word!r}: {e}")
                return False
        current.update(code=code, tr=translators[code])
        print(f"Captions: English -> {language_name(code)}")
        return True

    if translate_to and not set_language(translate_to):
        sys.exit(1)
    label.setup(mode=mode)
    print("Loading Whisper...", file=sys.stderr)
    whisper = Transcriber(model_size)  # same settings the accuracy test (wer_test) uses

    q: queue.Queue = queue.Queue()
    recording = threading.Event()
    quitting = threading.Event()
    captured = [0]  # total samples captured while recording (for the lag warning)

    def on_audio(indata, frames, t, status):
        if recording.is_set():
            q.put(indata[:, 0].copy())
            captured[0] += frames

    def read_commands():
        for line in sys.stdin:
            parts = line.strip().lower().split(maxsplit=1)
            cmd = parts[0] if parts else ""
            if cmd == "start" and len(parts) == 2:  # "start spanish"
                if not set_language(parts[1]):
                    continue
                if not recording.is_set():
                    recording.set()
                    print(">> Recording... (type stop to pause)")
            elif cmd in ("start", "") and not recording.is_set():
                recording.set()
                print(">> Recording... (type stop to pause)")
            elif cmd in ("stop", "") and recording.is_set():
                recording.clear()
                print("|| Stopped. Type start to resume or quit to exit.")
            elif cmd in ("quit", "exit", "q"):
                recording.clear()
                quitting.set()
                return
            elif cmd not in ("start", "stop"):
                print("Commands: start, start <language>, stop, quit")
        quitting.set()  # stdin closed

    chunker = PauseChunker(pause=pause_seconds, max_seconds=chunk_seconds)

    def transcribe(start_sample, chunk):
        offset = start_sample / SAMPLE_RATE  # where this utterance sits in the recording
        for seg in whisper(chunk):
            a = chunk[int(seg.start * SAMPLE_RATE):int(seg.end * SAMPLE_RATE)]
            speaker, text = label.label_segment(
                a, seg.text.strip(), offset + seg.start, offset + seg.end)
            if text:
                print(f"[{offset + seg.start:6.1f}s] {speaker:>10}: {highlight_terms(text)}")
                code, tr = current["code"], current["tr"]
                if tr:
                    print(f"{'':>8} {code:>10}: {tr(text)}")

    last_warn = 0.0

    def warn_if_behind(start_sample, chunk):
        nonlocal last_warn
        lag = (captured[0] - start_sample - len(chunk)) / SAMPLE_RATE  # audio not yet captioned
        if lag > 2 * chunk_seconds and time.monotonic() - last_warn > 10:
            last_warn = time.monotonic()
            print(f"(running {lag:.1f}s behind: plug in the laptop, or try --model tiny.en)",
                  file=sys.stderr)

    threading.Thread(target=read_commands, daemon=True).start()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        callback=on_audio, device=device):
        print(f"\nReady (mode={label._mode}). Type start, or e.g. start spanish, "
              "and press Enter to begin recording.")
        try:
            while not quitting.is_set():
                try:
                    block = q.get(timeout=0.2)
                except queue.Empty:
                    # Stopped and everything captured has arrived: transcribe the tail.
                    if not recording.is_set():
                        for start, chunk in chunker.flush():
                            transcribe(start, chunk)
                    continue
                # Each utterance goes to Whisper as soon as the speaker pauses.
                for start, chunk in chunker.feed(block):
                    transcribe(start, chunk)
                    warn_if_behind(start, chunk)
        except KeyboardInterrupt:
            recording.clear()
        tail = []
        while not q.empty():
            tail += chunker.feed(q.get())
        for start, chunk in tail + chunker.flush():
            transcribe(start, chunk)
    print("Bye.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="*", help="'enroll <name>' or nothing for live captions")
    p.add_argument("--mode", choices=["pause", "embed"], default="pause",
                   help="speaker labels: pause = new speaker after a long pause (default); "
                        "embed = by voice, needs 'enroll' recordings first")
    p.add_argument("--chunk", type=float, default=6.0,
                   help="longest stretch of speech (s) before it's cut between words")
    p.add_argument("--pause", type=float, default=0.5,
                   help="silence (s) that ends an utterance and sends it to Whisper")
    p.add_argument("--model", default="base")
    p.add_argument("--device", type=int, default=None, help="mic index from --list-devices")
    p.add_argument("--list-devices", action="store_true")
    p.add_argument("--translate", metavar="LANG", default=None,
                   help="also show captions translated from English to LANG (e.g. es, hi, zh)")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    elif args.command[:1] == ["enroll"] and len(args.command) == 2:
        record_enrollment(args.command[1], args.device)
    elif not args.command:
        live(args.mode, args.chunk, args.model, args.device, args.translate, args.pause)
    else:
        p.error("usage: live_test [enroll NAME] [--mode embed]")


if __name__ == "__main__":
    main()
