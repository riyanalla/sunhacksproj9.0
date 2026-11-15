"""Quiz generation, translation, and slide text extraction."""

import json
import os
import re
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import requests

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_SLIDE_CONTEXT = 6000


def _generate_text(prompt: str, max_output_tokens: int, json_response: bool = False) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY to use quiz and translation features.")

    generation_config = {
        "maxOutputTokens": max_output_tokens,
        "temperature": 0.2,
    }
    if json_response:
        generation_config["responseMimeType"] = "application/json"

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(MODEL, safe='')}:generateContent",
        headers={"x-goog-api-key": api_key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()

    candidates = payload.get("candidates", [])
    if not candidates:
        reason = payload.get("promptFeedback", {}).get("blockReason")
        detail = f" ({reason})" if reason else ""
        raise ValueError(f"Gemini returned no response text{detail}.")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text.strip():
        raise ValueError("Gemini returned no response text.")
    return text


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


def make_quiz(transcript: str, slides: str = "", avoid: list[str] | tuple = ()) -> dict:
    """Create one plain-language multiple-choice question from lecture text.

    avoid: questions already asked, so "Another question" asks about something new.
    """
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError("A non-empty lecture transcript is required to make a quiz.")

    slide_context = ""
    if slides and slides.strip():
        slide_context = (
            "Slides for terminology and context; use them to interpret the "
            "transcript, but do not ask about facts found only in the slides:\n"
            f"{slides[:MAX_SLIDE_CONTEXT]}\n\n"
        )

    prompt = (
        "Write one multiple-choice question that checks an important idea stated "
        "in the lecture transcript. The transcript is the primary source; use "
        "slides only to interpret terms and provide context. Do not ask about "
        "slide-only facts. Use plain, clear English and one idea per sentence. "
        "Avoid idioms and trick wording. Keep each answer option under 12 words.\n"
        "Write the question and explanation as direct statements about the subject. "
        "Never mention the source: do not write phrases like 'according to the "
        "transcript', 'according to the lecture', 'the professor said', or 'in the "
        "text'.\n"
        "Include exactly four options with exactly one correct answer. The three "
        "wrong options must be close calls a student could believe: about the same "
        "topic, using terms from this lecture, matching the correct answer's length, "
        "grammar and style, and based on common misconceptions or easily confused "
        "ideas (for example swapping two related concepts). No joke options, no "
        "options that are obviously off-topic, no 'all of the above' or 'none of "
        "the above'.\n"
        "Return only a JSON object with this shape: "
        '{"question": string, "options": [four strings], '
        '"answer": integer index from 0 to 3, "explanation": string}.\n\n'
        + (
            "These questions were already asked. Ask about a different idea, or the "
            "same idea from a clearly different angle:\n- " + "\n- ".join(avoid) + "\n\n"
            if avoid else ""
        )
        + f"{slide_context}Lecture transcript:\n{transcript}"
    )
    quiz = _parse_quiz(_generate_text(prompt, max_output_tokens=400, json_response=True))
    return _shuffle_options(_drop_source_mentions(quiz))


# Safety net for the prompt rule above: "According to the transcript, X" -> "X".
_SOURCE_PREFIX = re.compile(
    r"^\s*(?:according to|based on|as (?:stated|mentioned|described|explained) in|in)\s+"
    r"(?:the\s+)?(?:lecture\s+)?(?:transcript|lecture|text|professor|slides?|passage)"
    r"(?:'s\s+\w+)?\s*,\s*",
    re.IGNORECASE,
)


def _drop_source_mentions(quiz: dict) -> dict:
    out = dict(quiz)
    for key in ("question", "explanation"):
        text = _SOURCE_PREFIX.sub("", out[key])
        out[key] = text[:1].upper() + text[1:]
    return out


def _shuffle_options(quiz: dict) -> dict:
    """Language models tend to put the right answer first; shuffle so it can be anywhere."""
    import random
    order = list(range(4))
    random.shuffle(order)
    return {**quiz, "options": [quiz["options"][i] for i in order],
            "answer": order.index(quiz["answer"])}


def translate(text: str, lang: str) -> str:
    """Translate text into the requested language, preserving its meaning."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Non-empty text is required for translation.")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("A target language is required for translation.")

    prompt = (
        f"Translate the text into {lang.strip()}. Preserve its meaning "
        "and return only the translation:\n\n"
        f"{text}"
    )
    return _generate_text(prompt, max_output_tokens=500).strip()


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