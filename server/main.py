# server/main.py
from collections import defaultdict
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
import db
from middleware import configure_middleware
from quiz import extract_slide_text

# Create the FastAPI application and attach shared HTTP middleware.
app = FastAPI()
configure_middleware(app)

# used to keep track of which connected clients are watching which lecture sessions
rooms: dict[str, set[WebSocket]] = defaultdict(set)

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
    
@app.get("/sessions")
def sessions():
    # Return the MongoDB IDs of all lecture sessions.
    return db.list_sessions()

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

    return {
        "slide_id": slide_id,
        "session_id": session_id,
        "filename": file.filename,
        "extracted_characters": len(extracted_text),
    }

@app.get("/sessions/{session_id}/slides")
def slides(session_id: str):
    try:
        return {"session_id": session_id, "text": db.get_slides(session_id)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/search")
def search_route(q: str, session_id: str | None = None):
    # Search all transcripts, or restrict the search to one lecture session.
    try:
        return db.search(q, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc