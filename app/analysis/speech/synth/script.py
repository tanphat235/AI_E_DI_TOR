"""Reading a narration script.

Pure functions over strings: no TTS, no files beyond a read, no settings object. So the
splitting rules below are testable directly, which matters because **a script line becomes a
narration beat, and a beat is what footage gets matched against.** Split too finely and the
director is asked to find a shot for a three-word fragment; split too coarsely and one clip
has to cover fifteen seconds of speech.

## The format

Plain text. One paragraph per beat, blank line between beats:

    Đầu tiên, chuẩn bị đất cho thật tơi.

    Sau đó tưới nước cho kỹ, đừng để đọng.

    Cuối cùng, phủ một lớp mùn quanh gốc.

That is the whole format. Two conveniences on top, because a script is written by a person:

* **A single newline inside a paragraph is a soft wrap**, not a new beat. Editors wrap; the
  writer did not mean a beat boundary at column 80.
* **Lines starting with ``#`` are comments** and are not spoken. A script needs somewhere to
  put a stage direction without it being read aloud.

When a paragraph is long enough that no single shot could cover it, it is split at sentence
boundaries. The threshold is config, not a constant here, because it is really a statement
about how long you are willing to hold one shot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.utils.logging import get_logger

logger = get_logger(__name__)

SCRIPT_VERSION = "script/1"

_COMMENT = re.compile(r"^\s*#")
# The fullwidth punctuation below is deliberate, not a typo: a script pasted out of a CJK
# editor really does contain those characters, and a splitter that ignored them would treat
# such a script as one enormous paragraph. Hence the RUF001 suppression on the line itself.
_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])\s+")  # noqa: RUF001
"""Split after terminal punctuation followed by whitespace.

Includes the CJK and ellipsis forms because a Vietnamese script routinely carries ``…`` and
occasionally full-width punctuation pasted from elsewhere. A lookbehind rather than a capture
so the punctuation stays attached to the sentence it ends.
"""

_ABBREVIATIONS = ("Mr.", "Mrs.", "Ms.", "Dr.", "St.", "vs.", "etc.", "e.g.", "i.e.", "TS.", "ThS.")
"""Do not split after these, even though they end in a period.

A short list on purpose. Sentence segmentation is a genuinely hard problem and this is not
attempting to solve it — these are the cases common enough that getting them wrong would be
noticed, and the format's real answer to ambiguity is a blank line.
"""


class ScriptError(ValueError):
    """Raised when a script file is unusable - missing, unreadable, or empty of speech."""


@dataclass(frozen=True, slots=True)
class ScriptLine:
    """One beat of narration, as written."""

    index: int
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def parse_script(text: str, *, max_words_per_line: int = 0) -> tuple[ScriptLine, ...]:
    """Split a script into beats.

    Args:
        text: The script's contents.
        max_words_per_line: Split a paragraph at sentence boundaries once it exceeds this
            many words. ``0`` disables splitting, so a paragraph is always exactly one beat.

    Raises:
        ScriptError: nothing speakable was found.
    """
    paragraphs = _paragraphs(text)
    if not paragraphs:
        msg = "the script contains no speakable text (blank, or only comment lines starting with #)"
        raise ScriptError(msg)

    lines: list[str] = []
    for paragraph in paragraphs:
        lines.extend(_split_if_long(paragraph, max_words=max_words_per_line))

    logger.debug("Parsed %d paragraph(s) into %d beat(s)", len(paragraphs), len(lines))
    return tuple(ScriptLine(index=index, text=line) for index, line in enumerate(lines))


def load_script(path: Path, *, max_words_per_line: int = 0) -> tuple[ScriptLine, ...]:
    """Read and parse a script file.

    Raises:
        ScriptError: the file is missing, unreadable, or holds nothing speakable.
    """
    if not path.is_file():
        msg = f"no script at {path}"
        raise ScriptError(msg)
    try:
        # utf-8-sig: a script written in Notepad carries a BOM, and an unstripped BOM
        # becomes an invisible first character that the voice tries to pronounce.
        content = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        msg = f"could not read {path.name}: {exc}"
        raise ScriptError(msg) from exc
    return parse_script(content, max_words_per_line=max_words_per_line)


def _paragraphs(text: str) -> list[str]:
    """Blank-line-separated blocks, comments dropped, soft wraps joined."""
    blocks: list[str] = []
    current: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if _COMMENT.match(raw):
            continue
        if not line:
            if current:
                blocks.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        blocks.append(" ".join(current))
    return [block for block in (_normalise(item) for item in blocks) if block]


def _normalise(text: str) -> str:
    """Collapse runs of whitespace.

    A double space between sentences is invisible in a text editor and audible as a stumble
    in some voices, so it is removed rather than passed through.
    """
    return re.sub(r"\s+", " ", text).strip()


def _split_if_long(paragraph: str, *, max_words: int) -> list[str]:
    """Split a paragraph at sentence boundaries if it exceeds ``max_words``.

    Sentences are recombined greedily rather than emitted one per beat: splitting a
    three-sentence paragraph into three beats when two would fit under the limit makes the
    edit cut more than the writing asked for.
    """
    if max_words <= 0 or len(paragraph.split()) <= max_words:
        return [paragraph]

    sentences = _sentences(paragraph)
    if len(sentences) == 1:
        # One very long sentence. Splitting mid-sentence would put a cut inside a clause,
        # which is worse than one long shot, so it is left alone and the caller sees a beat
        # over the limit.
        logger.debug("A single sentence exceeds %d words; leaving it whole", max_words)
        return [paragraph]

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = [*current, sentence]
        if current and len(" ".join(candidate).split()) > max_words:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current = candidate
    if current:
        chunks.append(" ".join(current))
    return chunks


def _sentences(paragraph: str) -> list[str]:
    """Split into sentences, keeping known abbreviations intact."""
    # The abbreviations are masked rather than handled by a cleverer regex: a lookbehind that
    # excludes them would have to enumerate them anyway, and this way the list is the only
    # thing to edit.
    masked = paragraph
    for position, abbreviation in enumerate(_ABBREVIATIONS):
        masked = masked.replace(abbreviation, f"\x00{position}\x00")

    parts = [part.strip() for part in _SENTENCE_END.split(masked) if part.strip()]

    restored: list[str] = []
    for part in parts:
        for position, abbreviation in enumerate(_ABBREVIATIONS):
            part = part.replace(f"\x00{position}\x00", abbreviation)
        restored.append(part)
    return restored or [paragraph]


__all__ = ["SCRIPT_VERSION", "ScriptError", "ScriptLine", "load_script", "parse_script"]
