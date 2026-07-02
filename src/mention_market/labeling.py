"""Ground-truth labeling: did the target phrase actually appear in a transcript?

The market resolves on a human reading of the speech, so our label needs to be
robust to the incidental variation a transcriber introduces — capitalization,
punctuation, curly vs straight quotes, hyphenation, and minor spacing. It must
*not* be so loose that it matches unrelated text.

Strategy: normalize both the transcript and each phrase form (lowercase, strip
punctuation to spaces, collapse whitespace), then test for a whitespace-bounded
substring match. Config supplies explicit ``variants`` for genuine rephrasings
(e.g. "witch hunt" / "witch-hunt"); we do not attempt open-ended paraphrase
matching, which would trade false negatives for unauditable false positives.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .schema import Phrase, Transcript

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, replace punctuation with spaces, collapse spaces.

    NFKC folding maps curly quotes/dashes and full-width characters to canonical
    forms before punctuation stripping, so "don't" and "don’t" normalize alike.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def phrase_appears(text: str, phrase: Phrase) -> bool:
    """True if any canonical/variant form of ``phrase`` appears in ``text``.

    Matching is on normalized text with whitespace-boundary anchoring so
    "fake news" does not match "fake newsletter".
    """
    norm_text = f" {normalize(text)} "
    for form in phrase.all_forms():
        norm_form = normalize(form)
        if not norm_form:
            continue
        if f" {norm_form} " in norm_text:
            return True
    return False


@dataclass(frozen=True, slots=True)
class LabelResult:
    transcript_id: str
    phrase_id: str
    label: int
    matched_form: str | None  # which form matched, for auditability


def label_transcript(transcript: Transcript, phrase: Phrase) -> LabelResult:
    """Produce a ground-truth label for one (transcript, phrase) pair."""
    norm_text = f" {normalize(transcript.text)} "
    matched: str | None = None
    for form in phrase.all_forms():
        norm_form = normalize(form)
        if norm_form and f" {norm_form} " in norm_text:
            matched = form
            break
    return LabelResult(
        transcript_id=transcript.transcript_id,
        phrase_id=phrase.phrase_id,
        label=int(matched is not None),
        matched_form=matched,
    )
