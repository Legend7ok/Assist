"""The full session on disk, for a review after the meeting.

The context window forgets what the summary absorbed; this file keeps everything that was said
and answered, one JSON record per line, in the order it happened.
"""

import asyncio
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

import structlog
from pydantic import JsonValue

from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDone,
    AnswerFailed,
    AnswerRequested,
    ComponentState,
    Event,
    NotesChanged,
    SegmentEdited,
    StatusChanged,
    SummaryEdited,
    SummaryUpdated,
    TranscriptFinal,
)
from assist.settings import StorageSettings
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
    StatusErrorRecord,
    SummaryEditedRecord,
    SummaryRecord,
)

log = structlog.get_logger(__name__)


class SessionRecorder:
    """Writes the session to `<sessions_dir>/<start date_time>.jsonl` as it happens.

    One subscription to every event keeps the records in the order things happened. Each line
    reaches the disk at once, so a crash loses nothing already written, and the disk is used
    off the event loop, so a slow disk does not hold up answers. Meeting notes are written
    only when they changed since the last record of them: before each answer and at the end.

    Start it before the bus, so the first events are not missed, and stop it after the bus has
    closed, so the events still queued are written too.
    """

    def __init__(
        self,
        bus: EventBus,
        settings: StorageSettings,
        *,
        clock: Callable[[], float],
        info: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self._settings = settings
        # Seconds since the session started, the clock of phrases and answers.
        self._clock = clock
        self._info = dict(info or {})
        self._path: Path | None = None
        self._file: TextIO | None = None
        # stop() writes its records and closes the file while an event may be being written.
        self._lock = asyncio.Lock()
        self._notes = ""
        self._notes_written = ""
        bus.subscribe(Event, self._on_event, name="session_recorder")

    @property
    def path(self) -> Path | None:
        return self._path

    async def start(self) -> Path:
        if self._path is not None:
            raise RuntimeError("Session recorder is already started")
        # The wall-clock moment of the session clock's zero: record times turn into the time
        # of day from it.
        started_at = datetime.now().astimezone() - timedelta(seconds=self._clock())
        path = self._settings.sessions_dir / f"{started_at:%Y-%m-%d_%H-%M-%S}.jsonl"
        async with self._lock:
            self._file = await asyncio.to_thread(_create, path)
            self._path = path
            await self._write(SessionStarted(started_at=started_at, info=self._info))
        log.info("session_recording_started", path=str(path))
        return path

    async def stop(self) -> Path | None:
        """Finish the file and export it to Markdown next to it.

        Returns the Markdown file, or None when there was nothing to stop or the export
        failed. A failed export leaves the session file whole: scripts/export_session.py
        exports it again.
        """
        async with self._lock:
            if self._file is None or self._path is None:
                return None
            try:
                at = self._now()
                if (notes := self._notes_if_changed(at)) is not None:
                    await self._write(notes)
                await self._write(SessionEnded(at=at))
            finally:
                file, self._file = self._file, None
                await asyncio.to_thread(file.close)
        log.info("session_recording_stopped", path=str(self._path))

        try:
            markdown = await asyncio.to_thread(export_session, self._path)
        except Exception:
            log.exception("session_export_failed", path=str(self._path))
            return None
        log.info("session_exported", path=str(markdown))
        return markdown

    async def _on_event(self, event: Event) -> None:
        async with self._lock:
            # Before start() and after stop() there is no file to write to.
            if self._file is None:
                return
            record = self._record(event)
            if record is not None:
                await self._write(record)

    def _record(self, event: Event) -> Record | None:
        # Partial phrases, audio and answer pieces are left out: the final phrase and the
        # finished answer carry all of their text.
        match event:
            case TranscriptFinal():
                return PhraseRecord(segment=event.segment)
            case SegmentEdited():
                return PhraseEditedRecord(
                    at=self._now(), segment_id=event.segment_id, text=event.text
                )
            case AnswerRequested():
                return self._notes_if_changed(self._now())
            case AnswerDone():
                return AnswerRecord(answer=event.answer)
            case AnswerFailed():
                return AnswerFailedRecord(
                    at=self._now(), answer_id=event.answer_id, reason=event.reason
                )
            case SummaryUpdated():
                return SummaryRecord(
                    at=self._now(), text=event.text, absorbed_ids=event.absorbed_ids
                )
            case SummaryEdited():
                return SummaryEditedRecord(at=self._now(), text=event.text)
            case NotesChanged():
                # The notes field may send them on every keystroke: they are written only
                # when an answer is asked for, the moment they matter.
                self._notes = event.text
            case StatusChanged(state=ComponentState.ERROR):
                return StatusErrorRecord(
                    at=self._now(), component=event.component, detail=event.detail
                )
        return None

    def _notes_if_changed(self, at: float) -> NotesRecord | None:
        if self._notes == self._notes_written:
            return None
        self._notes_written = self._notes
        return NotesRecord(at=at, text=self._notes)

    def _now(self) -> float:
        return round(self._clock(), 3)

    async def _write(self, record: Record) -> None:
        assert self._file is not None
        line = record.model_dump_json() + "\n"
        await asyncio.to_thread(_append, self._file, line)


def _create(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    # "x": an existing session file is never overwritten. LF on Windows too: one record is one
    # line for any tool that reads the file.
    return path.open("x", encoding="utf-8", newline="\n")


def _append(file: TextIO, line: str) -> None:
    file.write(line)
    file.flush()
    # flush() only hands the line to Windows; fsync puts it on the disk, so it survives a
    # power cut or a system crash too.
    os.fsync(file.fileno())
