"""Download and cache the Whisper model now, while the WiFi is good.

    python -m nlp.download_model            # base
    python -m nlp.download_model small
"""

import sys

from faster_whisper import WhisperModel

if __name__ == "__main__":
    size = sys.argv[1] if len(sys.argv) > 1 else "base"
    WhisperModel(size, device="cpu", compute_type="int8")
    print(f"Whisper '{size}' is cached and ready.")
