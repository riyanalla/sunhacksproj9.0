import sqlite3

conn = sqlite3.connect("captions.db", check_same_thread=False)
conn.execute("""CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT, segment_id INTEGER, speaker TEXT,
    text TEXT, start REAL, end REAL)""")
conn.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS segments_fts
    USING fts5(text, content='segments', content_rowid='id')""")
conn.execute("""CREATE TABLE IF NOT EXISTS slides (
    session_id TEXT PRIMARY KEY,
    text TEXT NOT NULL)""")
conn.commit()

def save_segment(m):
    cur = conn.execute(
        "INSERT INTO segments (session_id, segment_id, speaker, text, start, end) VALUES (?,?,?,?,?,?)",
        (m["session_id"], m["segment_id"], m["speaker"], m["text"], m["start"], m["end"]))
    conn.execute("INSERT INTO segments_fts(rowid, text) VALUES (?, ?)", (cur.lastrowid, m["text"]))
    conn.commit()

def list_sessions():
    rows = conn.execute("SELECT DISTINCT session_id FROM segments").fetchall()
    return [r[0] for r in rows]

def get_transcript(session_id):
    rows = conn.execute(
        "SELECT segment_id, speaker, text, start, end FROM segments WHERE session_id=? ORDER BY start",
        (session_id,)).fetchall()
    return [dict(zip(["segment_id", "speaker", "text", "start", "end"], r)) for r in rows]

def save_slides(session_id, text):
    conn.execute(
        "INSERT OR REPLACE INTO slides (session_id, text) VALUES (?, ?)",
        (session_id, text))
    conn.commit()

def get_slides(session_id):
    row = conn.execute(
        "SELECT text FROM slides WHERE session_id=?",
        (session_id,)).fetchone()
    return row[0] if row else ""

def search(q, session_id=None):
    sql = """SELECT s.session_id, s.segment_id, s.speaker, s.text, s.start, s.end
             FROM segments_fts f JOIN segments s ON s.id = f.rowid
             WHERE segments_fts MATCH ?"""
    args = [q]
    if session_id:
        sql += " AND s.session_id = ?"
        args.append(session_id)
    rows = conn.execute(sql, args).fetchall()
    return [dict(zip(["session_id", "segment_id", "speaker", "text", "start", "end"], r)) for r in rows]