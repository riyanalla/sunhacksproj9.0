# tools/fake_emitter.py
"""Send canned captions to the server in a loop, to test the page without a mic.

    python tools/fake_emitter.py              # creates a new lecture and prints its link
    python tools/fake_emitter.py <session_id> # or post into an existing lecture
"""
import sys
import time

import requests

SERVER = "http://localhost:8000"
LINES = [
    ("Professor", "So today we're covering eigenvalue decomposition."),
    ("Professor", "Does that apply to non-square matrices? Good question."),
    ("Professor", "For those we use SVD instead."),
]

# Session ids are MongoDB ids, so ask the server for a new one unless one was given.
if len(sys.argv) > 1:
    SESSION = sys.argv[1]
else:
    SESSION = requests.post(f"{SERVER}/sessions", json={"title": "fake emitter"}).json()["session_id"]
print(f"Posting captions to {SERVER}/?s={SESSION}")
URL = f"{SERVER}/sessions/{SESSION}/caption"

seg = 0
while True:
    for speaker, text in LINES:
        seg += 1
        words = text.split()
        for i in range(2, len(words), 2):  # partials
            requests.post(URL, json={"type": "caption", "session_id": SESSION,
                "segment_id": seg, "speaker": speaker,
                "text": " ".join(words[:i]), "start": seg * 3.0,
                "end": seg * 3.0 + 2.5, "is_final": False})
            time.sleep(0.4)
        requests.post(URL, json={"type": "caption", "session_id": SESSION,
            "segment_id": seg, "speaker": speaker, "text": text,
            "start": seg * 3.0, "end": seg * 3.0 + 2.5, "is_final": True})
        time.sleep(1.5)
