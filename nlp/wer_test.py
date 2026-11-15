"""Measure caption accuracy: read a known text aloud, get the word error rate (WER).

Captions appear live while you read (same as live_test); when you stop, they're
compared word by word with the text you meant to say.

    python -m nlp.wer_test --model tiny.en          # paste your text, then read it aloud
    python -m nlp.wer_test --ref script.txt         # take the text from a file instead
    python -m nlp.wer_test --audio nlp/wer_runs/run_0001.wav --model base
                                                    # re-score a saved recording with
                                                    # different settings (no re-reading)
    python -m nlp.wer_test --history                # table of every run so far
    python -m nlp.wer_test --audio nlp/wer_runs/run_0004.wav --slides nlp/scripts/sample_slides.txt
                                                    # same, with lecture slides as context

Every recording is saved in nlp/wer_runs/ (git-ignored) next to the text you read, so
you can compare models and settings on exactly the same audio. Transcription uses the
same chunking and Whisper settings as live_test.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path(__file__).with_name("wer_runs")
HISTORY = RUNS_DIR / "history.csv"
SAMPLE_RATE = 16000
SESSION = "wer-test"  # session id the --slides context is stored under


def ask(prompt: str) -> str:
    """input() that shares sys.stdin's buffer, so pasted lines are never lost."""
    print(prompt, end="", flush=True)
    return sys.stdin.readline()


def read_pasted_text() -> str:
    print("Paste the text you're going to read, then press Enter on an empty line.")
    print("(Blank lines end the paste, so paste one paragraph, or use --ref file.txt.)")
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line:  # stdin closed
            break
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines)


def record_live(tr, pause: float, max_seconds: float, device=None):
    """Record from the mic and print captions live, exactly like live_test.

    Returns (audio, segments, seconds spent transcribing).
    """
    import queue
    import threading

    import numpy as np
    import sounddevice as sd

    from . import context
    from .chunker import PauseChunker
    from .vocab import fix_terms, highlight_terms

    terms = context.fix_terms_for(tr.session_id)

    q: queue.Queue = queue.Queue()
    blocks, segments = [], []
    busy = [0.0]
    stop = threading.Event()
    chunker = PauseChunker(pause=pause, max_seconds=max_seconds)

    def caption(start, chunk):
        t = time.perf_counter()
        for seg in tr(chunk):
            s, e, text = start / SAMPLE_RATE + seg.start, start / SAMPLE_RATE + seg.end, seg.text.strip()
            segments.append((s, e, text))
            fixed = fix_terms(text, extra_terms=terms)
            print(f"  [{s:5.1f}s] {highlight_terms(fixed, extra_terms=terms)}", flush=True)
        busy[0] += time.perf_counter() - t

    ask("\nPress Enter, then read the text aloud. Press Enter again when you're done. ")
    threading.Thread(target=lambda: (sys.stdin.readline(), stop.set()), daemon=True).start()
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device,
                        callback=lambda d, f, t, s: q.put(d[:, 0].copy())):
        print("Recording... live captions below (Enter to stop)\n", flush=True)
        while not stop.is_set():
            try:
                block = q.get(timeout=0.1)
            except queue.Empty:
                continue
            blocks.append(block)
            for start, chunk in chunker.feed(block):
                caption(start, chunk)
    while not q.empty():  # audio that arrived just before Enter
        block = q.get()
        blocks.append(block)
        for start, chunk in chunker.feed(block):
            caption(start, chunk)
    for start, chunk in chunker.flush():
        caption(start, chunk)
    audio = np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
    print(f"\nRecorded {len(audio) / SAMPLE_RATE:.1f} s.")
    return audio, segments, busy[0]


def next_run_path() -> Path:
    RUNS_DIR.mkdir(exist_ok=True)
    n = 1
    while (RUNS_DIR / f"run_{n:04d}.wav").exists():
        n += 1
    return RUNS_DIR / f"run_{n:04d}.wav"


def show_history() -> None:
    if not HISTORY.exists():
        print("No runs yet.")
        return
    with open(HISTORY, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{'when':16} {'recording':12} {'model':9} {'pause':>5} {'chunk':>5} "
          f"{'words':>5} {'WER':>6} {'WER+fix':>7} {'speed':>6}")
    for r in rows:
        print(f"{r['when']:16} {r['recording']:12} {r['model']:9} {r['pause']:>5} "
              f"{r['chunk']:>5} {r['words']:>5} {float(r['wer']):6.1%} "
              f"{float(r['wer_fixed']):7.1%} {float(r['speed']):5.2f}x")
    print("\nWER+fix = after course-term fixing. speed = audio seconds per second of "
          "transcribing (above 1x keeps up live).")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", default="base")
    p.add_argument("--pause", type=float, default=0.5, help="same as live_test --pause")
    p.add_argument("--chunk", type=float, default=6.0, help="same as live_test --chunk")
    p.add_argument("--ref", help="text file with what you'll read (default: paste it)")
    p.add_argument("--audio", help="re-score a saved recording instead of recording")
    p.add_argument("--device", type=int, default=None, help="mic index (see live_test --list-devices)")
    p.add_argument("--slides", help="lecture slides/notes (.txt or .pdf) to use as context")
    p.add_argument("--history", action="store_true", help="show every past run and exit")
    args = p.parse_args()

    if args.history:
        return show_history()

    from faster_whisper import decode_audio
    import soundfile as sf

    from . import context
    from .asr import Transcriber, transcribe_audio
    from .vocab import fix_terms
    from .wer import score

    session = None
    if args.slides:
        session = SESSION
        info = context.set_context(session, context.read_slides(args.slides))
        print(f"Slide terms: {', '.join(info['hints']) or '(none new)'}", file=sys.stderr)

    # 1. Reference text + audio
    if args.audio:
        wav = Path(args.audio)
        audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
        saved_ref = wav.with_suffix(".txt")
        if args.ref:
            reference = Path(args.ref).read_text(encoding="utf-8")
        elif saved_ref.exists():
            reference = saved_ref.read_text(encoding="utf-8")
        else:
            reference = read_pasted_text()
    else:
        reference = Path(args.ref).read_text(encoding="utf-8") if args.ref else read_pasted_text()
        if not reference.strip():
            sys.exit("No text given.")
        print("Loading Whisper...", file=sys.stderr)
        tr = Transcriber(args.model, session_id=session)
        # 2a. Live captions while you read; the score is for exactly these captions
        audio, segments, elapsed = record_live(tr, args.pause, args.chunk, args.device)
        if len(audio) < SAMPLE_RATE // 2:
            sys.exit("Recording is empty or too short. Check the mic with live_test --list-devices.")
        wav = next_run_path()
        sf.write(wav, audio, SAMPLE_RATE)
        wav.with_suffix(".txt").write_text(reference.strip() + "\n", encoding="utf-8")

    # 2b. Re-scoring a saved recording: run it through the same chunking as live mode
    if args.audio:
        if len(audio) < SAMPLE_RATE // 2:
            sys.exit("Recording is empty or too short.")
        print("Loading Whisper...", file=sys.stderr)
        tr = Transcriber(args.model, session_id=session)
        print("Transcribing...", file=sys.stderr)
        t = time.perf_counter()
        segments = transcribe_audio(tr, audio, pause=args.pause, max_seconds=args.chunk)
        elapsed = time.perf_counter() - t
    raw = " ".join(text for _, _, text in segments)
    terms = context.fix_terms_for(session)
    fixed = " ".join(fix_terms(text, extra_terms=terms) for _, _, text in segments)

    # 3. Score
    r_raw, r = score(reference, raw), score(reference, fixed)
    speed = (len(audio) / SAMPLE_RATE) / max(elapsed, 1e-6)
    print(f"\nRecording : {wav}")
    label = args.model + ("+slides" if args.slides else "")
    print(f"Model     : {label}   (pause {args.pause}s, chunk {args.chunk}s)")
    print(f"\nYou said  : {reference.strip()}")
    print(f"Captions  : {fixed}")
    print(f"\nWord error rate: {r.wer:.1%}  ({r.substitutions} wrong, {r.deletions} missed, "
          f"{r.insertions} extra, out of {r.ref_words} words)")
    if abs(r_raw.wer - r.wer) > 1e-9:
        print(f"  (before course-term fixing: {r_raw.wer:.1%})")
    print(f"Speed: {speed:.2f}x real time ({elapsed:.1f} s to transcribe "
          f"{len(audio) / SAMPLE_RATE:.1f} s of audio)")
    if r.errors:
        print("\nMistakes:")
        for e in r.errors:
            print(f"  {e}")
        print(f"\nMarked up: {r.diff}")
    else:
        print("\nNo mistakes.")

    # 4. Log the run so different settings can be compared later
    new = not HISTORY.exists()
    RUNS_DIR.mkdir(exist_ok=True)
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["when", "recording", "model", "pause", "chunk", "words",
                        "wer", "wer_fixed", "speed", "transcript"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), wav.stem, label,
                    args.pause, args.chunk, r.ref_words, f"{r_raw.wer:.4f}", f"{r.wer:.4f}",
                    f"{speed:.2f}", fixed])
    print(f"\nLogged to {HISTORY}  (see all runs: python -m nlp.wer_test --history)")


if __name__ == "__main__":
    main()
