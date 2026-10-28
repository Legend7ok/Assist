from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from assist.core.schemas import Answer, Segment, Speaker
from assist.storage.export import export_session
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
)

STARTED = SessionStarted(
    started_at=datetime(2026, 10, 28, 14, 5, 33, tzinfo=timezone(timedelta(hours=2))),
    info={"languages": ["uk", "en"]},
)


def segment(segment_id: str, speaker: Speaker, text: str, start: float, end: float):
    return PhraseRecord(
        segment=Segment(id=segment_id, speaker=speaker, text=text, start=start, end=end)
    )


def write_session(folder: Path, *records: Record, tail: bytes = b"") -> Path:
    path = folder / "session.jsonl"
    lines = b"".join(record.model_dump_json().encode() + b"\n" for record in records)
    path.write_bytes(lines + tail)
    return path


def export(path: Path) -> str:
    markdown = export_session(path)
    assert markdown == path.with_suffix(".md")
    return markdown.read_text(encoding="utf-8")


def test_markdown_of_a_whole_session(tmp_path):
    path = write_session(
        tmp_path,
        STARTED,
        # The short reply ends first, so it is written before the longer phrase it follows.
        segment("s2", Speaker.ME, "Sure", 12, 13),
        segment("s1", Speaker.THEM, "Tell me about indexes", 10, 14),
        NotesRecord(at=15, text="Python interview"),
        AnswerRecord(
            answer=Answer(
                id="a1", text="- B-tree\n- Hash", at=15.5, request="Review:\n```py\nx = 1\n```"
            )
        ),
        AnswerFailedRecord(at=20, answer_id="a2", reason="provider did not respond in time"),
        SummaryRecord(at=70.2, text="- Talked about indexes", absorbed_ids=("s1", "s2")),
        NotesRecord(at=80, text="Python interview, senior"),
        SessionEnded(at=3725),
    )

    assert export(path) == (
        "# Session 2026-10-28 14:05\n"
        "\n"
        "- Started: 2026-10-28 14:05:33\n"
        "- Duration: 1:02:05\n"
        "- languages: uk, en\n"
        "\n"
        "## Meeting notes\n"
        "\n"
        "Python interview, senior\n"
        "\n"
        "## Timeline\n"
        "\n"
        "**[00:10] Interlocutor:** Tell me about indexes\n"
        "\n"
        "**[00:12] Me:** Sure\n"
        "\n"
        "### [00:15] Prompter answer\n"
        "\n"
        "**Request:**\n"
        "\n"
        "````text\n"
        "Review:\n"
        "```py\n"
        "x = 1\n"
        "```\n"
        "````\n"
        "\n"
        "- B-tree\n"
        "- Hash\n"
        "\n"
        "**[00:20] Answer failed:** provider did not respond in time\n"
        "\n"
        "### [01:10] Summary\n"
        "\n"
        "- Talked about indexes\n"
    )


def test_edited_phrase_shows_its_latest_text(tmp_path):
    path = write_session(
        tmp_path,
        STARTED,
        segment("s1", Speaker.THEM, "Use a indx", 0, 2),
        PhraseEditedRecord(at=5, segment_id="s1", text="Use a index"),
        PhraseEditedRecord(at=6, segment_id="s1", text="Use an index"),
        SessionEnded(at=10),
    )

    markdown = export(path)

    assert "**[00:00] Interlocutor:** Use an index\n" in markdown
    assert "indx" not in markdown


def test_edited_summary_stands_at_the_time_of_the_edit(tmp_path):
    path = write_session(
        tmp_path,
        STARTED,
        SummaryRecord(at=60, text="Talked about indx", absorbed_ids=()),
        segment("s1", Speaker.THEM, "Next question", 70, 72),
        SummaryEditedRecord(at=75, text="Talked about indexes"),
        SessionEnded(at=80),
    )

    markdown = export(path)

    assert markdown.endswith(
        "### [01:00] Summary\n\nTalked about indx\n\n"
        "**[01:10] Interlocutor:** Next question\n\n"
        "### [01:15] Summary edited\n\nTalked about indexes\n"
    )


def test_cut_off_last_line_is_skipped(tmp_path):
    cut = segment("s2", Speaker.ME, "Індекс прискорює пошук", 20, 25).model_dump_json().encode()
    # A crash in the middle of a write: the line ends inside a Cyrillic letter.
    tail = cut[: cut.index("Індекс".encode()) + 3]
    path = write_session(
        tmp_path, STARTED, segment("s1", Speaker.THEM, "Що таке індекс?", 10, 14), tail=tail
    )

    markdown = export(path)

    assert "**[00:10] Interlocutor:** Що таке індекс?" in markdown
    assert "- Duration: 0:00:14 (the recording was cut off)" in markdown


def test_broken_line_in_the_middle_of_the_file_is_an_error(tmp_path):
    path = write_session(tmp_path, STARTED, tail=b'{"type": "phr\n')
    path.write_bytes(path.read_bytes() + SessionEnded(at=10).model_dump_json().encode() + b"\n")

    with pytest.raises(ValidationError):
        export_session(path)


def test_file_without_the_session_start_is_refused(tmp_path):
    path = write_session(tmp_path, segment("s1", Speaker.THEM, "Hello", 0, 1))

    with pytest.raises(ValueError, match="not a session file"):
        export_session(path)
