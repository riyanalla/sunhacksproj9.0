"""Quiz generation, translation, and slide text extraction."""

import json
import os
import re
from pathlib import Path
from typing import BinaryIO

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
MAX_SLIDE_CONTEXT = 6000


def _client():
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "Quiz and translation features require the 'anthropic' package."
        ) from exc
    return Anthropic()


def _message_text(response) -> str:
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return block.text
    raise ValueError("The language model response did not contain text.")


def _parse_quiz(response_text: str) -> dict:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", text, flags=re.IGNORECASE)

    try:
        quiz = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("The language model returned invalid quiz JSON.") from exc

    if not isinstance(quiz, dict):
        raise ValueError("The language model quiz response must be a JSON object.")
    question = quiz.get("question")
    options = quiz.get("options")
    answer = quiz.get("answer")
    explanation = quiz.get("explanation")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("The quiz response must include a non-empty question.")
    if (
        not isinstance(options, list)
        or len(options) != 4
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError("The quiz response must include exactly four text options.")
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < 4:
        raise ValueError("The quiz answer must be an option index from 0 to 3.")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("The quiz response must include a non-empty explanation.")

    return {
        "question": question.strip(),
        "options": [option.strip() for option in options],
        "answer": answer,
        "explanation": explanation.strip(),
    }


def make_quiz(transcript: str, slides: str = "") -> dict:
    """Create one plain-language multiple-choice question from lecture text."""
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("A non-empty lecture transcript is required to make a quiz.")

    slide_context = ""
    if slides and slides.strip():
        slide_context = (
            "Slides for terminology and spelling only; base the question on the "
            "lecture excerpt:\n"
            f"{slides[:MAX_SLIDE_CONTEXT]}\n\n"
        )

    prompt = (
        "Write one multiple-choice question that checks an important idea stated "
        "in the lecture excerpt. Use plain, clear English and one idea per "
        "sentence. Avoid idioms, trick wording, and unsupported facts. Keep each "
        "answer option under 12 words. Include exactly four plausible options "
        "with exactly one correct answer. Return only a JSON object with this "
        'shape: {"question": string, "options": [four strings], '
        '"answer": integer index from 0 to 3, "explanation": string}.\n\n'
        f"{slide_context}Lecture excerpt:\n{transcript}"
    )
    response = _client().messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return _parse_quiz(_message_text(response))


def translate(text: str, lang: str) -> str:
    """Translate text into the requested language, preserving its meaning."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Non-empty text is required for translation.")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("A target language is required for translation.")

    response = _client().messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Translate the text into {lang.strip()}. Preserve its meaning "
                    "and return only the translation:\n\n"
                    f"{text}"
                ),
            }
        ],
    )
    return _message_text(response).strip()


def extract_slide_text(file: str | Path | BinaryIO) -> str:
    """Extract selectable text from a PDF file or binary file object."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF slide extraction requires the 'pypdf' package."
        ) from exc

    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()