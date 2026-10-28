import asyncio
from collections.abc import AsyncIterator

import pytest

from assist.assistant.answerer import Answerer
from assist.context.notes import MeetingNotes
from assist.context.summarizer import Summarizer
from assist.context.window import ContextWindow
from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDelta,
    AnswerDone,
    AnswerFailed,
    AnswerRequested,
    Event,
    NotesChanged,
    RequestChanged,
    TranscriptFinal,
)
from assist.core.schemas import Answer, Message, Segment, Speaker
from assist.llm.base import LLMError
from assist.settings import AssistantSettings, ContextSettings

HANG = object()


class FakeLLM:
    """Plays one reply per call: pieces of text, an error after them, or hanging forever."""

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[Message], int]] = []

    async def stream(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        self.calls.append((messages, max_tokens))
        for part in self.replies.pop(0):
            if part is HANG:
                await asyncio.Event().wait()
            if isinstance(part, Exception):
                raise part
            yield part


class Meeting:
    def __init__(self, bus: EventBus, llm: FakeLLM) -> None:
        self.bus = bus
        self.llm = llm
        self.window = ContextWindow(bus, ContextSettings())
        self.notes = MeetingNotes(bus)
        self.answerer = Answerer(
            bus,
            llm,
            AssistantSettings(answer_max_tokens=100, request_max_tokens=900),
            window=self.window,
            summarizer=Summarizer(bus, llm, ContextSettings()),
            notes=self.notes,
            clock=lambda: 42.0,
        )
        self.events: list[Event] = []

        async def record(event: Event) -> None:
            self.events.append(event)

        bus.subscribe(Event, record)

    async def publish(self, *events: Event) -> None:
        for event in events:
            self.bus.publish(event)
        for _ in range(20):
            await asyncio.sleep(0)

    def of_type[E: Event](self, event_type: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, event_type)]

    def prompt(self, call: int = 0) -> str:
        return "\n".join(message.content for message in self.llm.calls[call][0])


@pytest.fixture
async def make_meeting():
    started: list[Meeting] = []

    async def factory(*replies: object) -> Meeting:
        meeting = Meeting(EventBus(), FakeLLM(*replies))
        await meeting.bus.start()
        started.append(meeting)
        return meeting

    yield factory
    for meeting in started:
        await meeting.answerer.close()
        await meeting.bus.close()


QUESTION = TranscriptFinal(
    segment=Segment(speaker=Speaker.THEM, text="What is an index?", start=30, end=33)
)


async def test_answer_streams_in_order_and_ends_with_the_full_text(make_meeting):
    meeting = await make_meeting(["A structure ", "that speeds ", "up lookups."])

    await meeting.publish(QUESTION, AnswerRequested())

    assert [delta.text for delta in meeting.of_type(AnswerDelta)] == [
        "A structure ",
        "that speeds ",
        "up lookups.",
    ]
    (done,) = meeting.of_type(AnswerDone)
    assert done.answer.text == "A structure that speeds up lookups."
    assert {delta.answer_id for delta in meeting.of_type(AnswerDelta)} == {done.answer.id}


async def test_finished_answer_joins_the_window_at_the_press_time(make_meeting):
    meeting = await make_meeting(["Speeds up lookups."])

    await meeting.publish(QUESTION, AnswerRequested())

    assert meeting.window.items[-1] == meeting.of_type(AnswerDone)[0].answer
    assert isinstance(meeting.window.items[-1], Answer)
    assert meeting.window.items[-1].at == 42.0


async def test_provider_error_gives_answer_failed_and_nothing_joins_the_window(make_meeting):
    meeting = await make_meeting(["Partial ", LLMError("provider rate limit reached")])

    await meeting.publish(QUESTION, AnswerRequested())

    (failed,) = meeting.of_type(AnswerFailed)
    assert failed.reason == "provider rate limit reached"
    assert meeting.of_type(AnswerDone) == []
    assert [item.text for item in meeting.window.items] == ["What is an index?"]


async def test_empty_answer_is_a_failure(make_meeting):
    meeting = await make_meeting(["  "])

    await meeting.publish(AnswerRequested())

    (failed,) = meeting.of_type(AnswerFailed)
    assert failed.reason == "model returned an empty answer"


async def test_notes_window_and_request_go_into_the_prompt(make_meeting):
    meeting = await make_meeting(["Done."])

    await meeting.publish(
        QUESTION,
        NotesChanged(text="Interview for a Python role"),
        RequestChanged(text="Review this code"),
        AnswerRequested(),
    )

    prompt = meeting.prompt()
    assert "Interview for a Python role" in prompt
    assert "Interlocutor: What is an index?" in prompt
    assert "Review this code" in prompt
    assert meeting.llm.calls[0][1] == 900


async def test_request_is_cleared_after_the_answer(make_meeting):
    meeting = await make_meeting(["Done."])

    await meeting.publish(RequestChanged(text="Review this code"), AnswerRequested())

    assert meeting.notes.request == ""
    assert meeting.of_type(RequestChanged)[-1] == RequestChanged(text="")


async def test_request_is_kept_when_the_answer_fails(make_meeting):
    meeting = await make_meeting([LLMError("connection failed")])

    await meeting.publish(RequestChanged(text="Review this code"), AnswerRequested())

    assert meeting.notes.request == "Review this code"


async def test_new_press_cancels_the_answer_being_written_and_starts_a_new_one(make_meeting):
    meeting = await make_meeting(["Old ", HANG], ["New answer."])
    await meeting.publish(AnswerRequested())

    await meeting.publish(AnswerRequested())

    (failed,) = meeting.of_type(AnswerFailed)
    (done,) = meeting.of_type(AnswerDone)
    assert failed.answer_id != done.answer.id
    assert failed.reason == "cancelled by a new request"
    assert done.answer.text == "New answer."
    assert len(meeting.llm.calls) == 2
