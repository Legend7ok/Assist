import asyncio
from collections.abc import AsyncIterator

import pytest

from assist.context.summarizer import Summarizer
from assist.context.window import ContextWindow
from assist.core.bus import EventBus
from assist.core.events import (
    Component,
    ComponentState,
    Event,
    StatusChanged,
    SummaryEdited,
    SummaryFailed,
    SummaryRequested,
    SummaryUpdated,
    TranscriptFinal,
)
from assist.core.schemas import Message, Segment, Speaker
from assist.llm.base import LLMError
from assist.settings import ContextSettings

SETTINGS = ContextSettings(window_seconds=300, compress_seconds=120, summary_max_words=50)


class FakeLLM:
    """Replies with the given texts or errors in turn and keeps every prompt it got."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.prompts: list[list[Message]] = []
        self.gate: asyncio.Event | None = None

    async def stream(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        self.prompts.append(messages)
        if self.gate is not None:
            await self.gate.wait()
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        for word in reply.split(" "):
            yield word + " "


class Meeting:
    """Window and summarizer on one running bus, with every event published on it."""

    def __init__(self, bus: EventBus, llm: FakeLLM) -> None:
        self.bus = bus
        self.llm = llm
        self.window = ContextWindow(bus, SETTINGS)
        self.summarizer = Summarizer(bus, llm, SETTINGS)
        self.events: list[Event] = []

        async def record(event: Event) -> None:
            self.events.append(event)

        bus.subscribe(Event, record)

    async def publish(self, *events: Event) -> None:
        for event in events:
            self.bus.publish(event)
        await settle()

    def of_type[E: Event](self, event_type: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, event_type)]

    def texts(self) -> list[str]:
        return [item.text for item in self.window.items]


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
async def make_meeting():
    buses: list[EventBus] = []

    async def factory(*replies: str | Exception) -> Meeting:
        bus = EventBus()
        meeting = Meeting(bus, FakeLLM(*replies))
        await bus.start()
        buses.append(bus)
        return meeting

    yield factory
    for bus in buses:
        await bus.close()


def phrase(start: float, end: float, text: str) -> TranscriptFinal:
    return TranscriptFinal(segment=Segment(speaker=Speaker.THEM, text=text, start=start, end=end))


# The last phrase brings the window to 300 s and hands "a" and "b" to the summary.
FIRST_FULL = (phrase(0, 50, "a"), phrase(50, 110, "b"), phrase(110, 130, "c"))
FIRST_TRIGGER = phrase(130, 300, "d")


async def test_summary_absorbs_the_part_and_it_leaves_the_window(make_meeting):
    meeting = await make_meeting("first summary")

    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    assert meeting.summarizer.summary == "first summary"
    assert meeting.texts() == ["c", "d"]
    (updated,) = meeting.of_type(SummaryUpdated)
    assert updated.text == "first summary"


async def test_summary_builds_on_the_previous_one(make_meeting):
    meeting = await make_meeting("first summary", "second summary")
    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    # "c" and "d" now span 110-300; this phrase brings the window to 300 s again.
    await meeting.publish(phrase(300, 410, "e"))

    second_prompt = meeting.llm.prompts[1][1].content
    assert "first summary" in second_prompt
    assert "Interlocutor: c" in second_prompt
    assert meeting.summarizer.summary == "second summary"


async def test_word_limit_from_config_is_in_the_instructions(make_meeting):
    meeting = await make_meeting("summary")

    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    assert "At most 50 words" in meeting.llm.prompts[0][0].content


async def test_failed_summary_keeps_every_phrase_in_the_window(make_meeting):
    meeting = await make_meeting(LLMError("provider did not respond in time"))

    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    assert meeting.texts() == ["a", "b", "c", "d"]
    assert meeting.summarizer.summary == ""
    (failed,) = meeting.of_type(SummaryFailed)
    assert failed.reason == "provider did not respond in time"


async def test_empty_reply_is_a_failure_not_an_empty_summary(make_meeting):
    meeting = await make_meeting("   ")

    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    assert meeting.texts() == ["a", "b", "c", "d"]
    assert meeting.of_type(SummaryUpdated) == []


async def test_status_shows_busy_then_done_or_error(make_meeting):
    meeting = await make_meeting("summary", LLMError("rate limit"))
    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)
    await meeting.publish(phrase(300, 410, "e"))

    states = [(s.component, s.state) for s in meeting.of_type(StatusChanged)]
    assert states == [
        (Component.LLM_SUMMARY, ComponentState.BUSY),
        (Component.LLM_SUMMARY, ComponentState.OK),
        (Component.LLM_SUMMARY, ComponentState.BUSY),
        (Component.LLM_SUMMARY, ComponentState.ERROR),
    ]


async def test_next_summary_is_built_on_the_edited_one(make_meeting):
    meeting = await make_meeting("first summary", "second summary")
    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    await meeting.publish(SummaryEdited(text="corrected summary"), phrase(300, 410, "e"))

    second_prompt = meeting.llm.prompts[1][1].content
    assert "corrected summary" in second_prompt
    assert "first summary" not in second_prompt


async def test_edit_while_summarizing_is_rejected(make_meeting):
    meeting = await make_meeting("summary")
    meeting.llm.gate = asyncio.Event()
    await meeting.publish(*FIRST_FULL, FIRST_TRIGGER)

    await meeting.publish(SummaryEdited(text="edited during summary"))
    meeting.llm.gate.set()
    await settle()

    assert meeting.of_type(SummaryRequested)
    assert meeting.summarizer.summary == "summary"
