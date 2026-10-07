import asyncio

import pytest

from assist.context.window import ContextWindow
from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDone,
    SegmentEdited,
    SummaryFailed,
    SummaryRequested,
    SummaryUpdated,
    TranscriptFinal,
)
from assist.core.schemas import Answer, Segment, Speaker
from assist.settings import ContextSettings

SETTINGS = ContextSettings(window_seconds=300, compress_seconds=120)


class Context:
    """A window on a running bus, with every summary request it made."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self.window = ContextWindow(bus, SETTINGS)
        self.requests: list[SummaryRequested] = []

        async def record(event: SummaryRequested) -> None:
            self.requests.append(event)

        bus.subscribe(SummaryRequested, record)

    async def publish(self, *events) -> None:
        for event in events:
            self.bus.publish(event)
        for _ in range(5):
            await asyncio.sleep(0)

    def texts(self) -> list[str]:
        return [item.text for item in self.window.items]


@pytest.fixture
async def context():
    bus = EventBus()
    context = Context(bus)
    await bus.start()
    yield context
    await bus.close()


def phrase(start: float, end: float, text: str, speaker: Speaker = Speaker.THEM):
    return TranscriptFinal(segment=Segment(speaker=speaker, text=text, start=start, end=end))


# Phrases from 0 to 300 s: the window reaches its limit on the last one.
FULL_WINDOW = (
    phrase(0, 50, "a"),
    phrase(50, 110, "b"),
    phrase(110, 130, "c"),
    phrase(130, 300, "d"),
)


async def test_phrases_and_answers_share_one_timeline(context):
    await context.publish(
        phrase(10, 20, "question"),
        phrase(5, 30, "long phrase finished later"),
        AnswerDone(answer=Answer(text="answer", at=25)),
    )

    assert context.texts() == ["long phrase finished later", "question", "answer"]


async def test_no_summary_before_the_window_is_full(context):
    await context.publish(*FULL_WINDOW[:3])

    assert context.requests == []


async def test_summary_part_ends_at_the_end_of_a_phrase(context):
    await context.publish(*FULL_WINDOW)

    (request,) = context.requests
    # "c" ends at 130, past the 120 s part, so it stays whole in the window.
    assert [item.text for item in request.items] == ["a", "b"]


async def test_single_phrase_longer_than_the_part_is_still_summarized(context):
    await context.publish(phrase(0, 200, "monologue"), phrase(200, 300, "next"))

    (request,) = context.requests
    assert [item.text for item in request.items] == ["monologue"]


async def test_window_keeps_everything_while_the_summary_is_written(context):
    await context.publish(*FULL_WINDOW, phrase(300, 310, "e"))

    assert context.texts() == ["a", "b", "c", "d", "e"]


async def test_no_second_summary_while_one_is_running(context):
    await context.publish(*FULL_WINDOW, phrase(300, 310, "e"), phrase(310, 320, "f"))

    assert len(context.requests) == 1


async def test_summary_removes_absorbed_items_by_id(context):
    await context.publish(*FULL_WINDOW)
    (request,) = context.requests

    await context.publish(
        SummaryUpdated(text="summary", absorbed_ids=tuple(item.id for item in request.items))
    )

    assert context.texts() == ["c", "d"]


async def test_failed_summary_keeps_the_part_and_retries_on_the_next_phrase(context):
    await context.publish(*FULL_WINDOW)

    await context.publish(SummaryFailed(reason="timeout"), phrase(300, 310, "e"))

    assert context.texts() == ["a", "b", "c", "d", "e"]
    first, second = context.requests
    assert second.items == first.items


async def test_edit_replaces_the_phrase_text(context):
    event = phrase(0, 10, "Use a indx")
    await context.publish(event)

    await context.publish(SegmentEdited(segment_id=event.segment.id, text="Use an index"))

    assert context.texts() == ["Use an index"]


async def test_edit_of_a_phrase_being_summarized_is_rejected(context):
    await context.publish(*FULL_WINDOW)
    being_summarized, outside = FULL_WINDOW[0].segment, FULL_WINDOW[2].segment

    await context.publish(
        SegmentEdited(segment_id=being_summarized.id, text="changed"),
        SegmentEdited(segment_id=outside.id, text="changed"),
    )

    assert context.texts() == ["a", "b", "changed", "d"]
