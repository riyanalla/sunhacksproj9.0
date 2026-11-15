# tools/fake_emitter.py
import time, requests

URL = "http://localhost:8000/sessions/lecture-1/caption"
LINES = [
    ("Speaker 1", "So today we're covering eigenvalue decomposition."),
    ("Speaker 2", "Professor, does that apply to non-square matrices?"),
    ("Speaker 1", "Good question. For those we use SVD instead."),
]

seg = 0
while True:
    for speaker, text in LINES:
        seg += 1
        words = text.split()
        for i in range(2, len(words), 2):  # partials
            requests.post(URL, json={"type": "caption", "session_id": "lecture-1",
                "segment_id": seg, "speaker": speaker,
                "text": " ".join(words[:i]), "start": seg * 3.0,
                "end": seg * 3.0 + 2.5, "is_final": False})
            time.sleep(0.4)
        requests.post(URL, json={"type": "caption", "session_id": "lecture-1",
            "segment_id": seg, "speaker": speaker, "text": text,
            "start": seg * 3.0, "end": seg * 3.0 + 2.5, "is_final": True})
        time.sleep(1.5)