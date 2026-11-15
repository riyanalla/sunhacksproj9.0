"""Offline English -> other language translation for captions (Argos Translate).

    from nlp.translate import Translator
    tr = Translator("es")          # downloads the en->es model the first time (~100 MB)
    tr("gradient descent is an optimization method")

Language codes: es, fr, de, hi, zh, ja, ko, ar, pt, it, ru, ... (run
`python -m nlp.translate --list` to see every target available from English).
Download the models you need ahead of time while the WiFi is good:

    python -m nlp.translate es hi
"""

from __future__ import annotations

import sys
import traceback

SOURCE = "en"

# Words people type after "start" -> Argos language code. Raw codes ("es") also work.
LANGUAGE_NAMES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "dutch": "nl", "russian": "ru", "ukrainian": "uk", "polish": "pl",
    "turkish": "tr", "arabic": "ar", "hebrew": "he", "persian": "fa", "farsi": "fa",
    "hindi": "hi", "urdu": "ur", "bengali": "bn", "chinese": "zh", "mandarin": "zh",
    "japanese": "ja", "korean": "ko", "vietnamese": "vi", "indonesian": "id",
    "thai": "th", "greek": "el", "swedish": "sv", "czech": "cs", "romanian": "ro",
}


def resolve_language(word: str) -> str:
    """'Spanish' or 'es' -> 'es'. Unknown words are returned as-is (treated as a code)."""
    word = word.strip().lower()
    return LANGUAGE_NAMES.get(word, word)


def language_name(code: str) -> str:
    for name, c in LANGUAGE_NAMES.items():
        if c == code:
            return name.capitalize()
    return code


def _available_targets() -> dict[str, object]:
    import argostranslate.package as pkg
    pkg.update_package_index()
    return {p.to_code: p for p in pkg.get_available_packages() if p.from_code == SOURCE}


def ensure_installed(target: str) -> None:
    """Download the en->target model if it isn't cached yet."""
    import argostranslate.translate as tr
    if _find(tr, target):
        return
    print(f"[nlp] downloading en->{target} translation model...", file=sys.stderr)
    targets = _available_targets()
    if target not in targets:
        raise ValueError(f"No English->{target!r} model. Available: {', '.join(sorted(targets))}")
    targets[target].install()


def _find(tr, target: str):
    langs = {l.code: l for l in tr.get_installed_languages()}
    if SOURCE in langs and target in langs:
        return langs[SOURCE].get_translation(langs[target])
    return None


class Translator:
    """Callable that turns English text into `target`. Never raises; returns the input on error."""

    def __init__(self, target: str):
        import argostranslate.translate as tr
        self.target = target
        ensure_installed(target)
        self._t = _find(tr, target)
        if self._t is None:
            raise RuntimeError(f"en->{target} model failed to load")
        self._t.translate("Warm up.")  # first call loads the sentence splitter (~5 s)

    def __call__(self, text: str) -> str:
        if not text.strip():
            return text
        try:
            return self._t.translate(text)
        except Exception:
            traceback.print_exc()
            return text


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args == ["--list"]:
        print("English ->", ", ".join(sorted(_available_targets())))
    else:
        for code in args:
            ensure_installed(code)
            print(f"en->{code} ready: {Translator(code)('Hello, welcome to the lecture.')}")
