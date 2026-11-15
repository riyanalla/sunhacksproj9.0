# server/main.py
import asyncio
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

from bson import ObjectId
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

import db
import quiz
from middleware import configure_middleware
from quiz import extract_slide_text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `import nlp` works when the server runs from server/

# Speech recognition settings (see nlp/README.md).
NLP_MODEL = os.getenv("NLP_MODEL", "base")
NLP_SPEAKER_MODE = os.getenv("NLP_SPEAKER_MODE", "single")  # one professor: "Professor"
QUIZ_WINDOW_SECONDS = 30   # quiz on the last 30 s of lecture...
QUIZ_MIN_WORDS = 40        # ...extended back until there's enough to ask about

# Create the FastAPI application and attach shared HTTP middleware.
app = FastAPI()
configure_middleware(app)

# used to keep track of which connected clients are watching which lecture sessions
rooms: dict[str, set[WebSocket]] = defaultdict(set)
# when each lecture's audio first started (seconds since epoch), for caption timestamps
lecture_started: dict[str, float] = {}
# last few quiz questions per lecture, so "Another question" doesn't repeat itself
recent_questions: dict[str, list[str]] = defaultdict(list)


# ---- Web page -----------------------------------------------------------------------

@app.get("/")
def index():
    # The page uses relative URLs, so it must be served by this server.
    return FileResponse(ROOT / "index.html")


@app.get("/signin")
def signin_page():
    return FileResponse(ROOT / "signin.html")


@app.post("/auth/signin")
def signin(body: dict):
    identifier = body.get("identifier")
    password = body.get("password")
    if not isinstance(identifier, str) or not isinstance(password, str):
        raise HTTPException(status_code=400, detail="identifier and password are required")
    user = db.authenticate_user(identifier, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid username/email or password")
    return {"ok": True, "user": user}


# Optional mascot art and fonts that index.html looks for next to itself (if present;
# without them the page falls back to its built-in drawings and fonts).
for _folder in ("images", "fonts"):
    if (ROOT / _folder).is_dir():
        app.mount(f"/{_folder}", StaticFiles(directory=ROOT / _folder), name=_folder)


# ---- Captions out -------------------------------------------------------------------

@app.websocket("/ws/captions/{session_id}")
async def captions(ws: WebSocket, session_id: str):
    # Accept the browser connection and add it to this lecture's room.
    await ws.accept()
    rooms[session_id].add(ws)
    try:
        while True:
            # Keep the connection open while the client waits for broadcasts.
            await ws.receive_text()
    except WebSocketDisconnect:
        # Remove disconnected clients so future broadcasts do not target them.
        rooms[session_id].discard(ws)

async def broadcast(session_id: str, msg: dict):
    # Send the caption or lifecycle event to every client watching this session.
    for ws in list(rooms[session_id]):
        try:
            await ws.send_json(msg)
        except Exception:
            # A failed send usually means the client disconnected.
            rooms[session_id].discard(ws)

# The ASR pipeline or fake emitter sends caption events through this endpoint.
@app.post("/sessions/{session_id}/caption")
async def post_caption(session_id: str, msg: dict):
    # Prevent a message from being written to a different session than its URL.
    if msg.get("session_id") != session_id:
        raise HTTPException(status_code=400, detail="session_id does not match URL")

    # These are the only event types understood by the caption protocol.
    if msg.get("type") not in {"caption", "session_start", "session_end"}:
        raise HTTPException(status_code=422, detail="unsupported message type")

    if msg.get("type") == "caption" and msg.get("is_final"):
        # Partial captions are broadcast only; final captions are persisted.
        required = ("segment_id", "speaker", "text", "start", "end")
        if any(field not in msg for field in required):
            raise HTTPException(status_code=422, detail="final caption is missing fields")
        try:
            db.save_segment(msg)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    await broadcast(session_id, msg)
    return {"ok": True}


# ---- Audio in: browser mic -> Whisper (nlp) -> captions -------------------------------

def _new_stream(session_id: str):
    """Set up speech recognition for one mic connection (slow the first time: loads Whisper)."""
    from nlp import context, enhance
    from nlp.stream import LiveStream

    enhance.setup(NLP_SPEAKER_MODE)
    if context.version(session_id) == 0:  # e.g. after a server restart: reload the slides
        slides_text = db.get_slides(session_id)
        if slides_text:
            context.set_context(session_id, slides_text)

    # Captions keep counting from where the lecture is, even if the mic reconnects.
    previous = db.get_transcript(session_id)
    last_end = previous[-1]["end"] + 1 if previous else 0.0
    started = lecture_started.get(session_id)
    offset = max(last_end, time.time() - started) if started else last_end
    lecture_started.setdefault(session_id, time.time() - offset)
    return LiveStream(session_id, model_size=NLP_MODEL, start_offset=offset)


async def _publish(session_id: str, caption: dict):
    """Save a final caption and send it to every phone watching this lecture."""
    try:
        await run_in_threadpool(db.save_segment, caption)
    except Exception:
        traceback.print_exc()  # still show it live even if saving failed
    await broadcast(session_id, caption)


async def _transcribe(queue: asyncio.Queue, stream, session_id: str):
    """Worker: feed queued audio to Whisper off the event loop, publish captions."""
    while True:
        data = await queue.get()
        done = data is None
        chunks = [] if done else [data]
        while not done and not queue.empty():  # catch up: take everything waiting
            item = queue.get_nowait()
            if item is None:
                done = True
            else:
                chunks.append(item)
        try:
            caps = await run_in_threadpool(stream.feed_pcm, b"".join(chunks)) if chunks else []
            if done:
                caps += await run_in_threadpool(stream.flush)
        except Exception:
            traceback.print_exc()
            caps = []
        for caption in caps:
            await _publish(session_id, caption)
        if done:
            return


@app.websocket("/ws/audio/{session_id}")
async def audio(ws: WebSocket, session_id: str):
    """Mic audio from the lecture laptop's browser: binary frames of 16 kHz mono int16 PCM
    (PROTOCOL.md). Send the text message "stop" before closing to end the lecture."""
    await ws.accept()
    if not ObjectId.is_valid(session_id):
        await ws.send_json({"type": "error", "detail": "unknown session; create one with POST /sessions"})
        await ws.close(code=1008)
        return
    first = session_id not in lecture_started
    try:
        stream = await run_in_threadpool(_new_stream, session_id)
    except Exception as exc:
        traceback.print_exc()
        await ws.send_json({"type": "error", "detail": f"speech recognition unavailable: {exc}"})
        await ws.close(code=1011)
        return
    await ws.send_json({"type": "ready"})
    if first:
        await broadcast(session_id, {"type": "session_start", "session_id": session_id})

    queue: asyncio.Queue = asyncio.Queue()
    worker = asyncio.create_task(_transcribe(queue, stream, session_id))
    ended = False
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                queue.put_nowait(msg["bytes"])
            elif msg.get("text") == "stop":
                ended = True
                break
    except WebSocketDisconnect:
        pass
    finally:
        queue.put_nowait(None)  # caption whatever is still buffered
        await worker
    if ended:
        await broadcast(session_id, {"type": "session_end", "session_id": session_id})
        try:
            await ws.close()
        except Exception:
            pass


# ---- Sessions, transcript, slides, quiz, search -------------------------------------

@app.post("/sessions")
def new_session(body: dict | None = None):
    # Create a lecture; its ID goes in the page link as ?s=<id>.
    return {"session_id": db.create_session((body or {}).get("title"))}

@app.get("/sessions")
def sessions():
    # Return the MongoDB IDs of all lecture sessions.
    return db.list_sessions()

@app.get("/sessions/latest")
def latest_session():
    # MongoDB IDs start with their creation time, so the largest is the newest.
    ids = [s for s in db.list_sessions() if ObjectId.is_valid(s)]
    return {"session_id": max(ids, key=ObjectId) if ids else None}

@app.get("/sessions/{session_id}/transcript")
def transcript(session_id: str):
    # Convert invalid MongoDB IDs into a client-facing validation error.
    try:
        return db.get_transcript(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/sessions/{session_id}/slides")
async def upload_slides(session_id: str, file: UploadFile = File(...)):
    # Extract selectable text before storing the slide metadata and content.
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=415, detail="only PDF slide files are supported")

    try:
        extracted_text = extract_slide_text(file.file)
        slide_id = db.save_slides(
            session_id,
            file.filename or "slides.pdf",
            file.content_type,
            extracted_text,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Captions for this lecture now use the slides' key terms (nlp/context.py).
    terms = []
    try:
        from nlp.context import set_context
        terms = set_context(session_id, extracted_text)["hints"]
    except Exception:
        traceback.print_exc()  # the upload itself still worked

    return {
        "slide_id": slide_id,
        "session_id": session_id,
        "filename": file.filename,
        "extracted_characters": len(extracted_text),
        "caption_terms": terms,
    }

@app.get("/sessions/{session_id}/slides")
def slides(session_id: str):
    try:
        return {"session_id": session_id, "text": db.get_slides(session_id)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def recent_transcript(segments: list[dict], seconds: float = QUIZ_WINDOW_SECONDS,
                      min_words: int = QUIZ_MIN_WORDS, max_words: int = 400) -> str:
    """Text of the last `seconds` of lecture, reaching further back if that's too short."""
    if not segments:
        return ""
    last_end = segments[-1]["end"]
    picked, words = [], 0
    for seg in reversed(segments):
        recent = last_end - seg["start"] <= seconds
        if not recent and words >= min_words or words >= max_words:
            break
        picked.append(seg["text"])
        words += len(seg["text"].split())
    return " ".join(reversed(picked))

@app.post("/sessions/{session_id}/quiz")
def session_quiz(session_id: str):
    # One multiple-choice question about the last 30 s of the lecture (plus slides).
    try:
        text = recent_transcript(db.get_transcript(session_id))
        slides_text = db.get_slides(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(text.split()) < 8:
        raise HTTPException(status_code=409, detail="not enough lecture transcript yet")
    try:
        asked = recent_questions[session_id]
        q = quiz.make_quiz(text, slides_text, avoid=asked[-5:])
        asked.append(q["question"])
        del asked[:-5]
        return q
    except RuntimeError as exc:  # no GEMINI_API_KEY
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:     # Gemini down, bad JSON, ...
        traceback.print_exc()
        raise HTTPException(status_code=502, detail=f"quiz generation failed: {exc}") from exc

@app.get("/search")
def search_route(q: str, session_id: str | None = None):
    # Search all transcripts, or restrict the search to one lecture session.
    try:
        return db.search(q, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
