from datetime import datetime, timedelta
from pathlib import Path

import pytest

from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDelta,
    AnswerDone,
    AnswerFailed,
    AnswerRequested,
    AudioFrame,
    Component,
    ComponentState,
    Event,
    NotesChanged,
    RequestChanged,
    SegmentEdited,
    StatusChanged,
    SummaryEdited,
    SummaryUpdated,
    TranscriptFinal,
    TranscriptPartial,
)
from assist.core.schemas import Answer, Segment, Speaker
from assist.settings import StorageSettings
from assist.storage.records import Record, record_adapter
from assist.storage.session import SessionRecorder

INFO = {"languages": ["uk", "en"]}


class Session:
    """A started recorder on a running bus, writing into a temporary folder."""

    def __init__(self, bus: EventBus, folder: Path) -> None:
        self.bus = bus
        self.now = 0.0
        self.recorder = SessionRecorder(
            bus, StorageSettings(sessions_dir=folder), clock=lambda: self.now, info=INFO
        )

    def publish(self, *events: Event) -> None:
        for event in events:
            self.bus.publish(event)

    async def finish(self) -> None:
        # Closing the bus first lets the recorder write everything still queued.
        await self.bus.close()
        await self.recorder.stop()

    def records(self) -> list[Record]:
        assert self.recorder.path is not None
        lines = self.recorder.path.read_text(encoding="utf-8").splitlines()
        return [record_adapter.validate_json(line) for line in lines]

    def types(self) -> list[str]:
        return [record.type for record in self.records()]


@pytest.fixture
async def session(tmp_path: Path):
    bus = EventBus()
    session = Session(bus, tmp_path)
    await session.recorder.start()
    await bus.start()
    yield session
    await session.finish()


def phrase(text: str, start: float = 0.0) -> TranscriptFinal:
    return TranscriptFinal(
        segment=Segment(speaker=Speaker.THEM, text=text, start=start, end=start + 2)
    )


async def test_everything_said_and_answered_is_written_in_order(session):
    question = phrase("What is an index?")
    answer = Answer(text="A structure for lookups.", at=5.0, request="Explain simply")

    session.publish(
        question,
        SegmentEdited(segment_id=question.segment.id, text="What is an index"),
        NotesChanged(text="Python interview"),
        AnswerRequested(),
        AnswerDone(answer=answer),
        AnswerFailed(answer_id="a2", reason="provider did not respond in time"),
        SummaryUpdated(text="Talked about indexes", absorbed_ids=(question.segment.id,)),
        SummaryEdited(text="Talked about DB indexes"),
        StatusChanged(component=Component.STT, state=ComponentState.ERROR, detail="cut off"),
    )
    await session.finish()

    records = session.records()
    assert [record.type for record in records] == [
        "session_started",
        "phrase",
        "phrase_edited",
        "notes",
        "answer",
        "answer_failed",
        "summary",
        "summary_edited",
        "status_error",
        "session_ended",
    ]
    assert records[1].segment == question.segment
    assert records[4].answer == answer
    assert records[6].absorbed_ids == (question.segment.id,)
    assert records[8].detail == "cut off"


async def test_start_record_has_the_time_of_the_session_start_and_the_given_info(tmp_path):
    session = Session(EventBus(), tmp_path)
    session.now = 30.0  # the recorder starts 30 s into the session

    path = await session.recorder.start()
    await session.recorder.stop()

    started = session.records()[0]
    assert started.type == "session_started"
    assert started.info == INFO
    expected = datetime.now().astimezone() - timedelta(seconds=30)
    assert abs(started.started_at - expected) < timedelta(seconds=5)
    assert path.name == f"{started.started_at:%Y-%m-%d_%H-%M-%S}.jsonl"
    assert path.parent == tmp_path


async def test_partials_audio_answer_pieces_and_routine_statuses_are_not_written(session):
    session.publish(
        AudioFrame(speaker=Speaker.ME, pcm=bytes(640), start=1.0),
        TranscriptPartial(speaker=Speaker.ME, text="What is"),
        AnswerDelta(answer_id="a1", text="A struct"),
        RequestChanged(text="Explain simply"),
        StatusChanged(component=Component.STT, state=ComponentState.OK),
    )
    await session.finish()

    assert session.types() == ["session_started", "session_ended"]


async def test_notes_are_written_before_an_answer_only_when_they_changed(session):
    session.publish(
        NotesChanged(text="a"),
        AnswerRequested(),
        AnswerRequested(),
        NotesChanged(text="b"),
        NotesChanged(text="c"),
        AnswerRequested(),
    )
    await session.finish()

    notes = [record.text for record in session.records() if record.type == "notes"]
    assert notes == ["a", "c"]


async def test_notes_changed_after_the_last_answer_are_written_at_the_end(session):
    session.publish(NotesChanged(text="a"), AnswerRequested(), NotesChanged(text="b"))
    await session.finish()

    records = session.records()
    assert [record.type for record in records[-2:]] == ["notes", "session_ended"]
    assert records[-2].text == "b"


async def test_each_record_is_in_the_file_before_the_session_stops(session):
    session.publish(phrase("first"), phrase("second", start=3))
    await session.bus.close()  # every event handled, the file still open

    assert [record.segment.text for record in session.records()[1:]] == ["first", "second"]


async def test_events_after_stop_are_not_written_and_second_stop_does_nothing(session):
    await session.recorder.stop()

    session.publish(phrase("too late"))
    await session.finish()

    assert session.types() == ["session_started", "session_ended"]


async def test_stop_exports_the_session_to_markdown_next_to_it(session):
    session.publish(phrase("What is an index?"))
    await session.bus.close()

    markdown = await session.recorder.stop()

    assert session.recorder.path is not None
    assert markdown == session.recorder.path.with_suffix(".md")
    assert "Interlocutor:** What is an index?" in markdown.read_text(encoding="utf-8")
    assert await session.recorder.stop() is None
