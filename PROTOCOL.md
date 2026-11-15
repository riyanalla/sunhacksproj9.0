# CaptionCast / LevelUp Live protocol

Session ids are MongoDB ids (24 hex characters), created with `POST /sessions`. The page
link is `http://<server>:8000/?s=<session_id>`; without `?s=` the page joins the newest
lecture.

## Audio in (lecture laptop's browser → server)

WebSocket at `/ws/audio/{session_id}`.

- Client sends **binary frames of raw PCM: 16 kHz, mono, 16-bit, about 100 ms per frame
  (3,200 bytes)**. The page resamples the mic to this itself.
- Server replies `{"type": "ready"}` once speech recognition is loaded, or
  `{"type": "error", "detail": "..."}` and closes.
- Client sends the text message `"stop"` to end the lecture (server finishes the last
  captions and broadcasts `session_end`). Closing without `"stop"` (e.g. Wi-Fi drop) keeps
  the lecture going; reconnecting continues it.

The server turns the audio into captions with `nlp.stream.LiveStream` (Whisper + course
vocabulary + slide terms) and handles them like any other final caption below.

## Captions out (server → phones)

WebSocket at `/ws/captions/{session_id}`. JSON messages:

```json
{ "type": "caption", "session_id": "6551f0c2...", "segment_id": 42, "speaker": "Professor",
  "text": "So the posterior distribution...", "start": 12.4, "end": 15.1, "is_final": true }
```

`is_final: false` = a partial result. The frontend replaces it in place when a final
message with the same `segment_id` arrives.

Other `type` values: `"session_start"` and `"session_end"`, each with only `session_id`.

**Translation.** The page sends `{"type": "lang", "lang": "es"}` on this socket when it
connects and whenever the student changes the language menu. Each final caption is sent
in English right away; if anyone in the lecture picked another language, the server then
sends the same caption again (same `segment_id`) with
`"translations": {"es": "...", "ko": "..."}` (one Gemini call for all languages in use).
The page replaces it in place. Needs `GEMINI_API_KEY`; without it captions stay English.

## REST (server)

| Endpoint | Does |
|---|---|
| `GET /` | The web page (`index.html`) |
| `POST /sessions` | New lecture → `{"session_id": ...}` (optional body `{"title": ...}`) |
| `GET /sessions` | List of session ids |
| `GET /sessions/latest` | `{"session_id": <newest or null>}` |
| `POST /sessions/{id}/caption` | Post a caption / `session_start` / `session_end` message (fake emitter, other ASR) |
| `GET /sessions/{id}/transcript` | All final segments for a session |
| `POST /sessions/{id}/slides` | Upload a PDF (form field `file`); text is stored and its key terms improve captions |
| `GET /sessions/{id}/slides` | Newest uploaded slide text |
| `POST /sessions/{id}/quiz` | One multiple-choice question (Gemini) from the last ~30 s of transcript + slides: `{"question", "options": [4], "answer": 0-3, "explanation"}`. 409 = not enough transcript yet, 503 = no `GEMINI_API_KEY` |
| `GET /search?q=...&session_id=...` | Matching segments (MongoDB text search) |

Fake emitter: `tools/fake_emitter.py` creates a lecture and posts canned captions in a
loop, so the page can be tested without a mic.
