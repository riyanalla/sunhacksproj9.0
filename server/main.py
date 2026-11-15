# server/main.py
from collections import defaultdict
from io import BytesIO
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import db
import quiz
from pypdf.errors import EmptyFileError, PdfReadError

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_SLIDES_PDF_BYTES = 20 * 1024 * 1024

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
    if msg.get("type") == "caption" and msg.get("is_final"):
        try:
            db.save_segment(msg)
        except Exception as e:
            print("DB save failed:", e)
    await broadcast(session_id, msg)
    return {"ok": True}
    
@app.get("/sessions")
def sessions():
    return db.list_sessions()

@app.get("/sessions/{session_id}/transcript")
def transcript(session_id: str):
    return db.get_transcript(session_id)

@app.post("/sessions/{session_id}/slides")
async def upload_slides(session_id: str, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Upload a PDF file.")

    content = await file.read(MAX_SLIDES_PDF_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(content) > MAX_SLIDES_PDF_BYTES:
        raise HTTPException(status_code=413, detail="The PDF must be 20 MB or smaller.")

    try:
        slide_text = quiz.extract_slide_text(BytesIO(content))
    except (EmptyFileError, PdfReadError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable PDF.") from exc

    db.save_slides(session_id, slide_text)
    return {"session_id": session_id, "characters": len(slide_text)}

@app.post("/sessions/{session_id}/quiz")
def make_session_quiz(session_id: str):
    segments = db.get_transcript(session_id)
    if not segments:
        raise HTTPException(status_code=404, detail="No transcript is available for this session.")

    transcript_text = "\n".join(
        f"{segment['speaker']}: {segment['text']}" if segment["speaker"] else segment["text"]
        for segment in segments
    )
    return quiz.make_quiz(transcript_text, db.get_slides(session_id))

@app.get("/search")
def search_route(q: str, session_id: str | None = None):
    return db.search(q, session_id)