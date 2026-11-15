# LevelUp Live (sunhacksproj9.0)

Live lecture captions with quizzes. A laptop in the lecture listens through its browser,
the server transcribes with Whisper, and every student's phone or laptop gets captions,
a transcript, and "Quiz me" questions about the last 30 seconds.

```
lecture laptop (browser mic, "Start listening")
   │  /ws/audio/{session}  16 kHz PCM
   ▼
server/ (FastAPI) ── nlp/ (Whisper, course vocabulary, slide terms) ── MongoDB (transcript, slides)
   │  /ws/captions/{session}                     │
   ▼                                             └─ quiz.py (Gemini) ← "Quiz me"
students' phones / laptops (index.html)
```

| Folder | What |
|---|---|
| `index.html` | The web page (served by the server at `/`) |
| `server/` | FastAPI server: sessions, captions, slides, quiz, search |
| `nlp/` | Speech recognition: Whisper, chunking, course vocabulary, slide terms (see `nlp/README.md`) |
| `tools/fake_emitter.py` | Canned captions, to test the page without a mic |
| `PROTOCOL.md` | Message formats and endpoints |

## Run it (Windows)

One Python environment runs everything. From the repo root:

```
python -m venv nlp\.venv
nlp\.venv\Scripts\activate
pip install -r nlp\requirements.txt -r server\requirements.txt
pip install resemblyzer --no-deps
python -m nlp.download_model base          # once, while online
```

Start the server (from `server/`):

```
cd server
set GEMINI_API_KEY=your-key-here           # optional: real quiz questions (otherwise the page makes simple ones)
uvicorn main:app --host 0.0.0.0 --port 8000
```

Then:

1. **Lecture laptop**: open **http://localhost:8000** on the laptop running the server and
   press **Start listening**. Allow the microphone. The first start takes a few seconds
   (loading Whisper). The link changes to `?s=<lecture id>`.
2. **Students**: open `http://<laptop's IP>:8000` on the same Wi-Fi (find the IP with
   `ipconfig`). Without `?s=` they join the newest lecture. Allow Python through Windows
   Firewall on private networks if phones can't connect.
3. **Slides**: "My slides" takes images (shown on that device) or a PDF (uploaded: used
   for quiz questions and to spell course terms right in the captions).
4. **Stop listening** ends the lecture for everyone.

The microphone only works on `http://localhost` (or https), so "Start listening" has to
be pressed on the laptop running the server.

Test without a mic: `python tools/fake_emitter.py` (prints a link to open).

## Tests

```
python -m pytest nlp                      # speech/vocabulary pieces
cd server && python -m pytest test_main.py   # server, with an in-memory database (no MongoDB)
```

## Settings (environment variables)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | (none) | Needed for server-made quiz questions |
| `NLP_MODEL` | `base` | Whisper model: `tiny.en` is faster but less accurate |
| `NLP_SPEAKER_MODE` | `single` | `single` labels everything "Professor"; `pause` guesses speaker changes from pauses |
| `MONGODB_URI`, `MONGODB_DATABASE` | see `server/db.py` | Database connection |
