import os
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ASCENDING, MongoClient


_mongo_uri = os.getenv("MONGODB_URI")
if not _mongo_uri:
    raise RuntimeError("MONGODB_URI must be set before starting the server")

_client = MongoClient(_mongo_uri)
_database = _client[os.getenv("MONGODB_DATABASE", "lecture_app")]
_sessions = _database["sessions"]
_segments = _database["transcript_segments"]

_segments.create_index(
    [("session_id", ASCENDING), ("segment_id", ASCENDING)],
    unique=True,
)
_segments.create_index([("session_id", ASCENDING), ("start_ms", ASCENDING)])
_segments.create_index([("text", "text")])


def _session_object_id(session_id: str) -> ObjectId:
    if not ObjectId.is_valid(session_id):
        raise ValueError("session_id must be a valid MongoDB ObjectId")
    return ObjectId(session_id)


def save_segment(message: dict) -> None:
    session_id = _session_object_id(message["session_id"])
    now = datetime.now(timezone.utc)
    start_ms = round(float(message["start"]) * 1000)
    end_ms = round(float(message["end"]) * 1000)

    _segments.update_one(
        {"session_id": session_id, "segment_id": int(message["segment_id"])},
        {
            "$set": {
                "text": message["text"],
                "speaker": message.get("speaker"),
                "start_ms": start_ms,
                "end_ms": end_ms,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )


def list_sessions() -> list[str]:
    return [str(session["_id"]) for session in _sessions.find({}, {"_id": 1})]


def get_transcript(session_id: str) -> list[dict]:
    object_id = _session_object_id(session_id)
    rows = _segments.find(
        {"session_id": object_id},
        {"_id": 0, "segment_id": 1, "speaker": 1, "text": 1, "start_ms": 1, "end_ms": 1},
    ).sort("start_ms", ASCENDING)

    return [
        {
            "segment_id": row["segment_id"],
            "speaker": row.get("speaker"),
            "text": row["text"],
            "start": row["start_ms"] / 1000,
            "end": row["end_ms"] / 1000,
        }
        for row in rows
    ]


def search(query: str, session_id: str | None = None) -> list[dict]:
    filters: dict = {"$text": {"$search": query}}
    if session_id:
        filters["session_id"] = _session_object_id(session_id)

    rows = _segments.find(
        filters,
        {
            "_id": 0,
            "session_id": 1,
            "segment_id": 1,
            "speaker": 1,
            "text": 1,
            "start_ms": 1,
            "end_ms": 1,
            "score": {"$meta": "textScore"},
        },
    ).sort([("score", {"$meta": "textScore"})])

    return [
        {
            "session_id": str(row["session_id"]),
            "segment_id": row["segment_id"],
            "speaker": row.get("speaker"),
            "text": row["text"],
            "start": row["start_ms"] / 1000,
            "end": row["end_ms"] / 1000,
        }
        for row in rows
    ]
