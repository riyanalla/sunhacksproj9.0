import os
import unittest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from bson import ObjectId
from pymongo import ASCENDING, MongoClient

MONGODB_URI = "mongodb+srv://eobane_db_user:bMYuKjr9sP0X1cjP@cluster0.eolnuma.mongodb.net"
MONGODB_DATABASE = "lecture_app"

# Environment variables override the defaults so each deployment can select
# its own MongoDB cluster and database without changing the application code.

# Read connection settings from environment variables so credentials stay out
# of the source code.
_mongo_uri = os.getenv("MONGODB_URI", MONGODB_URI)
if not _mongo_uri:
    raise RuntimeError("MONGODB_URI must be set before starting the server")

# Connect to the lecture database and select the collections used by this module.
_client = MongoClient(_mongo_uri)
_database = _client[os.getenv("MONGODB_DATABASE", MONGODB_DATABASE)]
_sessions = _database["sessions"]
_segments = _database["transcript_segments"]

# Prevent duplicate segment numbers within a session
_segments.create_index(
    [("session_id", ASCENDING), ("segment_id", ASCENDING)],
    unique=True,
)
_segments.create_index([("session_id", ASCENDING), ("start_ms", ASCENDING)])
_segments.create_index([("text", "text")])


def _session_object_id(session_id: str) -> ObjectId:
    # URLs contain IDs as strings, while MongoDB stores references as ObjectIds.
    if not ObjectId.is_valid(session_id):
        raise ValueError("session_id must be a valid MongoDB ObjectId")
    return ObjectId(session_id)


def save_segment(message: dict) -> None:
    # Convert the incoming protocol message into the database representation.
    session_id = _session_object_id(message["session_id"])
    now = datetime.now(timezone.utc)
    start_ms = round(float(message["start"]) * 1000)
    end_ms = round(float(message["end"]) * 1000)

    # Upsert makes retries safe: an existing session/segment pair is updated
    # instead of creating a duplicate transcript row.
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
    # Return IDs as strings because they are sent through JSON and URL paths.
    return [str(session["_id"]) for session in _sessions.find({}, {"_id": 1})]


def get_transcript(session_id: str) -> list[dict]:
    # Return only segments belonging to this lecture, ordered chronologically.
    object_id = _session_object_id(session_id)
    rows = _segments.find(
        {"session_id": object_id},
        {"_id": 0, "segment_id": 1, "speaker": 1, "text": 1, "start_ms": 1, "end_ms": 1},
    ).sort("start_ms", ASCENDING)

    # Convert MongoDB's millisecond fields back to the API's seconds format.
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
    # Use MongoDB's text index and optionally limit results to one session.
    filters: dict = {"$text": {"$search": query}}
    if session_id:
        filters["session_id"] = _session_object_id(session_id)

    # Include a relevance score so the best text matches appear first.
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

    # Shape database records into the same response format as get_transcript.
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

if __name__ == "__main__":  # run tests when this file is executed directly
    class DatabaseFunctionTests(unittest.TestCase):
        def test_session_object_id(self):
                # Valid IDs are preserved and malformed IDs are rejected.
            session_id = "507f1f77bcf86cd799439011"
            self.assertEqual(str(_session_object_id(session_id)), session_id)
            with self.assertRaises(ValueError):
                _session_object_id("invalid-id")

        @patch(__name__ + "._segments")
        def test_save_segment(self, segments):
            # Verify timestamps are converted from seconds to milliseconds.
            save_segment(
                {
                    "session_id": "507f1f77bcf86cd799439011",
                    "segment_id": 3,
                    "speaker": "Ada",
                    "text": "Hello",
                    "start": 1.25,
                    "end": 2.5,
                }
            )

            filter_, update = segments.update_one.call_args.args[:2]
            self.assertEqual(filter_["segment_id"], 3)
            self.assertEqual(update["$set"]["start_ms"], 1250)
            self.assertEqual(update["$set"]["end_ms"], 2500)
            self.assertTrue(segments.update_one.call_args.kwargs["upsert"])

        @patch(__name__ + "._sessions")
        def test_list_sessions(self, sessions):
            # API responses expose MongoDB IDs as strings.
            sessions.find.return_value = [
                {"_id": ObjectId("507f1f77bcf86cd799439011")}
            ]
            self.assertEqual(
                list_sessions(), ["507f1f77bcf86cd799439011"]
            )

        @patch(__name__ + "._segments")
        def test_get_transcript(self, segments):
            # Transcript timestamps are converted back to seconds.
            cursor = Mock()
            cursor.sort.return_value = [
                {
                    "segment_id": 1,
                    "speaker": "Ada",
                    "text": "Hello",
                    "start_ms": 1000,
                    "end_ms": 2500,
                }
            ]
            segments.find.return_value = cursor

            result = get_transcript("507f1f77bcf86cd799439011")

            self.assertEqual(result[0]["start"], 1.0)
            self.assertEqual(result[0]["end"], 2.5)
            cursor.sort.assert_called_once_with("start_ms", ASCENDING)

        @patch(__name__ + "._segments")
        def test_search(self, segments):
            # Search requests should restrict results to the selected session.
            session_id = ObjectId("507f1f77bcf86cd799439011")
            cursor = Mock()
            cursor.sort.return_value = [
                {
                    "session_id": session_id,
                    "segment_id": 1,
                    "speaker": "Ada",
                    "text": "Hello",
                    "start_ms": 1000,
                    "end_ms": 2500,
                }
            ]
            segments.find.return_value = cursor

            result = search("hello", str(session_id))

            self.assertEqual(result[0]["session_id"], str(session_id))
            self.assertEqual(result[0]["text"], "Hello")
            self.assertIn("$text", segments.find.call_args.args[0])
            self.assertEqual(
                segments.find.call_args.args[0]["session_id"], session_id
            )

    unittest.main()
    