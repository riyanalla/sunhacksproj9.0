"""Server tests: cd server && python -m pytest test_main.py

Uses an in-memory stand-in for db.py (never touches the real MongoDB) and a fake speech
recognizer, so it runs in seconds without Whisper.
"""

import sys
import types
import uuid

import pytest
from bson import ObjectId


class FakeDB(types.ModuleType):
    def __init__(self):
        super().__init__("db")
        self.reset()

    def reset(self):
        self.sessions, self.segments, self.slides = [], {}, {}
        self.users, self.auth_sessions = {}, {}

    def create_session(self, title=None):
        sid = str(ObjectId())
        self.sessions.append(sid)
        return sid

    def list_sessions(self):
        return list(self.sessions)

    def _check(self, sid):
        if not ObjectId.is_valid(sid):
            raise ValueError("session_id must be a valid MongoDB ObjectId")

    def save_segment(self, m):
        self._check(m["session_id"])
        self.segments.setdefault(m["session_id"], {})[int(m["segment_id"])] = {
            k: m[k] for k in ("segment_id", "speaker", "text", "start", "end")}

    def get_transcript(self, sid):
        self._check(sid)
        return sorted(self.segments.get(sid, {}).values(), key=lambda s: s["start"])

    def save_slides(self, sid, filename, file_type, text):
        self._check(sid)
        self.slides[sid] = text
        return str(ObjectId())

    def get_slides(self, sid):
        self._check(sid)
        return self.slides.get(sid, "")

    def search(self, q, sid=None):
        return []

    def authenticate_user(self, identifier, password):
        user = self.users.get(identifier.strip().lower())
        if user and user["password"] == password:
            return {k: user[k] for k in ("id", "username", "email", "xp", "level")}
        return None

    def create_auth_session(self, user_id):
        token = str(uuid.uuid4())
        self.auth_sessions[token] = user_id
        return token

    def get_user_for_auth_session(self, token):
        user_id = self.auth_sessions.get(token)
        return next(
            ({k: user[k] for k in ("id", "username", "email", "xp", "level")}
             for user in self.users.values() if user["id"] == user_id),
            None,
        )

    def save_user_progress(self, user_id, xp):
        user = next((user for user in self.users.values() if user["id"] == user_id), None)
        if not user:
            return None
        user["xp"], user["level"] = xp, xp // 50 + 1
        return {"xp": user["xp"], "level": user["level"]}


fake_db = FakeDB()
sys.modules["db"] = fake_db  # before main imports it

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(main.app)


@pytest.fixture(scope="module", autouse=True)
def one_event_loop():
    # Like the real server, every connection must share one event loop, or a broadcast
    # from the mic connection to a phone connection can stall.
    with client:
        yield


@pytest.fixture(autouse=True)
def clean():
    fake_db.reset()
    main.lecture_started.clear()
    main.rooms.clear()
    client.cookies.clear()


class FakeStream:
    """Stands in for nlp.stream.LiveStream: one caption per audio message."""

    def __init__(self, sid):
        self.sid, self.n, self.flushed = sid, 0, False

    def feed_pcm(self, pcm):
        self.n += 1
        return [{"type": "caption", "session_id": self.sid, "segment_id": self.n,
                 "speaker": "Professor", "text": f"heard {len(pcm)} bytes",
                 "start": float(self.n), "end": self.n + 0.5, "is_final": True}]

    def flush(self):
        self.flushed = True
        return []


def test_serves_the_page():
    r = client.get("/")
    assert r.status_code == 200 and "LevelUp Live" in r.text


def test_signin_verifies_credentials_from_database():
    fake_db.users["ada@example.com"] = {
        "id": "user-1", "username": "ada", "email": "ada@example.com",
        "password": "correct", "xp": 0, "level": 1,
    }
    assert client.get("/signin").status_code == 200
    signin = client.post("/auth/signin", json={
        "identifier": "ADA@EXAMPLE.COM", "password": "correct"
    })
    assert signin.json()["user"]["username"] == "ada"
    assert client.get("/auth/me").json()["xp"] == 0
    assert client.post("/auth/progress", json={"xp": 75}).json() == {"xp": 75, "level": 2}
    assert client.get("/auth/me").json()["level"] == 2
    assert client.post("/auth/signin", json={
        "identifier": "ada@example.com", "password": "wrong"
    }).status_code == 401


def test_create_session_and_latest():
    assert client.get("/sessions/latest").json() == {"session_id": None}
    a = client.post("/sessions", json={"title": "Lecture 1"}).json()["session_id"]
    b = client.post("/sessions").json()["session_id"]
    assert ObjectId.is_valid(a) and client.get("/sessions/latest").json()["session_id"] == b


def test_mic_audio_becomes_saved_and_broadcast_captions(monkeypatch):
    sid = client.post("/sessions").json()["session_id"]
    stream = FakeStream(sid)
    monkeypatch.setattr(main, "_new_stream", lambda s: stream)

    with client.websocket_connect(f"/ws/captions/{sid}") as phone:
        with client.websocket_connect(f"/ws/audio/{sid}") as mic:
            assert mic.receive_json() == {"type": "ready"}
            assert phone.receive_json()["type"] == "session_start"
            mic.send_bytes(b"\x00\x01" * 1600)
            cap = phone.receive_json()
            assert cap["type"] == "caption" and cap["text"] == "heard 3200 bytes"
            mic.send_text("stop")
        assert phone.receive_json() == {"type": "session_end", "session_id": sid}

    assert stream.flushed
    assert [s["text"] for s in client.get(f"/sessions/{sid}/transcript").json()] == \
        ["heard 3200 bytes"]


def test_mic_with_unknown_session_is_refused():
    with client.websocket_connect("/ws/audio/lecture-1") as mic:
        assert mic.receive_json()["type"] == "error"


def test_slides_upload_gives_captions_the_slide_terms(monkeypatch):
    from nlp import context

    sid = client.post("/sessions").json()["session_id"]
    text = "Gibbs sampling uses the full conditional.\nEach full conditional is easy."
    monkeypatch.setattr(main, "extract_slide_text", lambda f: text)
    r = client.post(f"/sessions/{sid}/slides",
                    files={"file": ("slides.pdf", b"%PDF-1.4", "application/pdf")})
    assert r.status_code == 200
    assert "full conditional" in r.json()["caption_terms"]
    assert "full conditional" in context.fix_terms_for(sid)
    assert client.get(f"/sessions/{sid}/slides").json()["text"] == text


def test_quiz_uses_recent_transcript_and_slides(monkeypatch):
    sid = client.post("/sessions").json()["session_id"]
    assert client.post(f"/sessions/{sid}/quiz").status_code == 409  # nothing said yet

    for i in range(1, 6):
        fake_db.save_segment({"session_id": sid, "segment_id": i, "speaker": "Professor",
                              "text": f"sentence number {i} about the posterior distribution",
                              "start": i * 10.0, "end": i * 10.0 + 5})
    fake_db.slides[sid] = "Slide text"
    seen = {}

    def fake_make_quiz(transcript, slides="", avoid=()):
        seen.update(transcript=transcript, slides=slides, avoid=list(avoid))
        return {"question": f"Q{len(avoid) + 1}?", "options": ["a", "b", "c", "d"],
                "answer": 1, "explanation": "because"}

    monkeypatch.setattr(main.quiz, "make_quiz", fake_make_quiz)
    main.recent_questions.clear()
    r = client.post(f"/sessions/{sid}/quiz")
    assert r.status_code == 200 and r.json()["answer"] == 1
    assert seen["slides"] == "Slide text" and "number 5" in seen["transcript"]
    assert seen["avoid"] == []
    client.post(f"/sessions/{sid}/quiz")           # "Another question"
    assert seen["avoid"] == ["Q1?"]                # told not to repeat the first one


def test_quiz_without_api_key_says_so(monkeypatch):
    sid = client.post("/sessions").json()["session_id"]
    fake_db.save_segment({"session_id": sid, "segment_id": 1, "speaker": "Professor",
                          "text": "a long enough sentence about Bayesian statistics today",
                          "start": 0.0, "end": 4.0})

    def no_key(*a, **k):
        raise RuntimeError("Set GEMINI_API_KEY to use quiz and translation features.")

    monkeypatch.setattr(main.quiz, "make_quiz", no_key)
    assert client.post(f"/sessions/{sid}/quiz").status_code == 503


def test_captions_are_translated_for_languages_in_use(monkeypatch):
    sid = client.post("/sessions").json()["session_id"]
    asked = []

    def fake_translate(text, langs):
        asked.append(sorted(langs))
        return {l: f"[{l}] {text}" for l in langs}

    monkeypatch.setattr(main.quiz, "translate_many", fake_translate)
    cap = {"type": "caption", "session_id": sid, "segment_id": 1, "speaker": "Professor",
           "text": "Hello class", "start": 0.0, "end": 1.0, "is_final": True}
    with client.websocket_connect(f"/ws/captions/{sid}") as phone_es, \
            client.websocket_connect(f"/ws/captions/{sid}") as phone_en:
        phone_es.send_text('{"type": "lang", "lang": "es"}')
        phone_en.send_text('{"type": "lang", "lang": "en"}')
        phone_en.send_text("not json")             # ignored, connection stays open
        client.post(f"/sessions/{sid}/caption", json=cap)
        first = phone_es.receive_json()
        assert first["text"] == "Hello class" and "translations" not in first  # English, no delay
        second = phone_es.receive_json()
        assert second["segment_id"] == 1 and second["translations"] == {"es": "[es] Hello class"}
    assert asked == [["es"]]                         # one call, only languages in use


def test_no_translation_when_everyone_reads_english(monkeypatch):
    sid = client.post("/sessions").json()["session_id"]
    monkeypatch.setattr(main.quiz, "translate_many",
                        lambda *a: pytest.fail("should not translate"))
    cap = {"type": "caption", "session_id": sid, "segment_id": 1, "speaker": "Professor",
           "text": "Hello", "start": 0.0, "end": 1.0, "is_final": True}
    with client.websocket_connect(f"/ws/captions/{sid}") as phone:
        client.post(f"/sessions/{sid}/caption", json=cap)
        assert phone.receive_json()["text"] == "Hello"


def test_recent_transcript_window():
    segs = [{"text": "old " * 50, "start": 0, "end": 10},
            {"text": "middle words here", "start": 60, "end": 65},
            {"text": "latest words", "start": 80, "end": 85}]
    # last 30 s is only 5 words, so it reaches back until there are 40+
    assert main.recent_transcript(segs).startswith("old")
    assert main.recent_transcript(segs, min_words=3) == "middle words here latest words"
