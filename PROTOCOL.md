Audio in (Pi → laptop):
WebSocket at
/ws/audio
. Binary frames of raw PCM: 16 kHz,mono, 16-bit, about 100 ms per frame (3,200 bytes).
Captions out (server → phones):
WebSocket at
/ws/captions/{session_id}
. JSONmessages:

{ "type": "caption", "session_id": "lecture-1", "segment_id": 42, "speaker": "Speaker 2", "text": "So the eigenvalue decomposition...", "start": 12.4, "end": 15.1, "is_final": true}
is_final: false
= a partial result. The frontend replaces it in place when a finalmessage with the same
segment_id
arrives.
Other
type
values:
"session_start"
and
"session_end"
, each with only
session_id
.
REST (server):

Endpoint
Returns
GET /sessions
List of saved sessions
GET /sessions/{id}/transcript
All final segments for a session
GET /search?q=...&session_id=...
Matching segments (SQLite FTS5)
POST /sessions/{id}/vocab
Upload course terms for the vocabulary system
Fake emitter: tools/fake_emitter.py sends canned caption messages on a loop, so thefrontend and server can be built before the ASR works.