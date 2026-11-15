# server/main.py
from collections import defaultdict
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import db

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

rooms: dict[str, set[WebSocket]] = defaultdict(set)

@app.websocket("/ws/captions/{session_id}")
async def captions(ws: WebSocket, session_id: str):
    await ws.accept()
    rooms[session_id].add(ws)
    try:
        while True:
            await ws.receive_text()  # keepalive; ignore content
    except WebSocketDisconnect:
        rooms[session_id].discard(ws)

async def broadcast(session_id: str, msg: dict):
    for ws in list(rooms[session_id]):
        try:
            await ws.send_json(msg)
        except Exception:
            rooms[session_id].discard(ws)

# Internal: fake emitter and the ASR pipeline push captions through here
@app.post("/sessions/{session_id}/caption")
async def post_caption(session_id: str, msg: dict):
    if msg.get("session_id") != session_id:
        raise HTTPException(status_code=400, detail="session_id does not match URL")

    if msg.get("type") not in {"caption", "session_start", "session_end"}:
        raise HTTPException(status_code=422, detail="unsupported message type")

    if msg.get("type") == "caption" and msg.get("is_final"):
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
    return db.list_sessions()

@app.get("/sessions/{session_id}/transcript")
def transcript(session_id: str):
    try:
        return db.get_transcript(session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/search")
def search_route(q: str, session_id: str | None = None):
    try:
        return db.search(q, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc