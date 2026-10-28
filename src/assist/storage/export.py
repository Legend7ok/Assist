"""A session file as Markdown, for reading after the meeting."""

import json
import re
from collections.abc import Sequence
from pathlib import Path

import structlog
from pydantic import JsonValue, ValidationError

from assist.core.schemas import Answer, Speaker
from assist.storage.records import (
    AnswerFailedRecord,
    AnswerRecord,
    NotesRecord,
    PhraseEditedRecord,
    PhraseRecord,
    Record,
    SessionEnded,
    SessionStarted,
    SummaryEditedRecord,
    SummaryRecord,
    record_adapter,
)

log = structlog.get_logger(__name__)

_LABELS = {Speaker.THEM: "Interlocutor", Speaker.ME: "Me"}


def export_session(path: Path) -> Path:
    """Write the session file `path` as Markdown into a `.md` next to it; return that file."""
    markdown = render_markdown(read_records(path))
    target = path.with_suffix(".md")
    target.write_text(markdown, encoding="utf-8", newline="\n")
    return target


def read_records(path: Path) -> list[Record]:
    # Bytes, not text: a crash can cut the last line in the middle of a Cyrillic letter, and
    # decoding the whole file would fail on it.
    lines = path.read_bytes().splitlines()
    records: list[Record] = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(record_adapter.validate_json(line))
        except ValidationError:
            # Every line reaches the disk on its own, so only the line being written when the
            # app crashed can be cut off, and all before it are whole.
            if number == len(lines):
                log.warning("session_last_line_cut_off", path=str(path), line=number)
                break
            raise
    return records


def render_markdown(records: Sequence[Record]) -> str:
    """Header, final meeting notes and the timeline of phrases, answers and summaries.

    Phrases show their latest edited text. Everything goes by its time in the session, so a
    long phrase that finished after a shorter one still stands where it started.
    """
    if not records or not isinstance(records[0], SessionStarted):
        raise ValueError("not a session file: it does not begin with the session start")
    started, rest = records[0], records[1:]

    edits = {r.segment_id: r.text for r in rest if isinstance(r, PhraseEditedRecord)}
    notes = ""
    ended: SessionEnded | None = None
    timeline: list[tuple[float, str]] = []
    for record in rest:
        match record:
            case PhraseRecord(segment=segment):
                text = edits.get(segment.id, segment.text)
                label = _LABELS[segment.speaker]
                timeline.append((segment.start, f"**{_clock(segment.start)} {label}:** {text}"))
            case AnswerRecord(answer=answer):
                timeline.append((answer.at, _answer(answer)))
            case AnswerFailedRecord():
                timeline.append(
                    (record.at, f"**{_clock(record.at)} Answer failed:** {record.reason}")
                )
            case SummaryRecord():
                timeline.append((record.at, f"### {_clock(record.at)} Summary\n\n{record.text}"))
            case SummaryEditedRecord():
                timeline.append(
                    (record.at, f"### {_clock(record.at)} Summary edited\n\n{record.text}")
                )
            case NotesRecord():
                notes = record.text
            case SessionEnded():
                ended = record
    # Stable: entries at the same moment keep the order they happened in.
    timeline.sort(key=lambda entry: entry[0])

    if ended is not None:
        duration = _duration(ended.at)
    else:
        # No end record: the app crashed, and the last thing written is as far as it got.
        last = max((_moment(record) for record in rest), default=0.0)
        duration = f"{_duration(last)} (the recording was cut off)"
    header = [
        f"# Session {started.started_at:%Y-%m-%d %H:%M}",
        "",
        f"- Started: {started.started_at:%Y-%m-%d %H:%M:%S}",
        f"- Duration: {duration}",
        *(f"- {key}: {_info_value(value)}" for key, value in started.info.items()),
    ]
    parts = ["\n".join(header)]
    if notes.strip():
        parts.append(f"## Meeting notes\n\n{notes.strip()}")
    parts.append("## Timeline")
    parts.extend(entry for _, entry in timeline)
    if not timeline:
        parts.append("Nothing was said.")
    return "\n\n".join(parts) + "\n"


def _answer(answer: Answer) -> str:
    lines = [f"### {_clock(answer.at)} Prompter answer"]
    if answer.request.strip():
        lines.append(f"**Request:**\n\n{_fenced(answer.request.strip())}")
    lines.append(answer.text)
    return "\n\n".join(lines)


def _fenced(text: str) -> str:
    # A request is often pasted code that may hold its own ``` blocks: the fence has to be
    # longer than any backtick run inside, or the inner one would close it.
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _moment(record: Record) -> float:
    match record:
        case PhraseRecord():
            return record.segment.end
        case AnswerRecord():
            return record.answer.at
        case SessionStarted():
            return 0.0
        case _:
            return record.at


def _clock(seconds: float) -> str:
    whole = int(seconds)
    return f"[{whole // 60:02d}:{whole % 60:02d}]"


def _duration(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def _info_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str | int | float) for item in value):
        return ", ".join(str(item) for item in value)
    return json.dumps(value, ensure_ascii=False)
