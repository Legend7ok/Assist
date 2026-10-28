import bisect
from itertools import takewhile

import structlog

from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDone,
    Event,
    SegmentEdited,
    SummaryFailed,
    SummaryRequested,
    SummaryUpdated,
    TranscriptFinal,
)
from assist.core.schemas import Answer, Segment, Speaker
from assist.settings import ContextSettings

log = structlog.get_logger(__name__)

type ContextItem = Segment | Answer


def _start(item: ContextItem) -> float:
    return item.start if isinstance(item, Segment) else item.at


def _end(item: ContextItem) -> float:
    return item.end if isinstance(item, Segment) else item.at


_LABELS = {Speaker.THEM: "Interlocutor", Speaker.ME: "Me"}


def format_timeline(items: tuple[ContextItem, ...]) -> str:
    """Items as `[mm:ss] Who: text` lines, the form both the answer and the summary see."""
    lines = []
    for item in items:
        seconds = int(_start(item))
        label = _LABELS[item.speaker] if isinstance(item, Segment) else "Prompter answer"
        lines.append(f"[{seconds // 60:02d}:{seconds % 60:02d}] {label}: {item.text}")
    return "\n".join(lines)


class ContextWindow:
    """Recent phrases and answers in time order; they go into the prompt word for word.

    When the window spans `window_seconds`, its oldest `compress_seconds` are handed to the
    summary with SummaryRequested. They stay in the window, and so in the prompt, until
    SummaryUpdated removes them, so nothing drops out of the context while the summary is
    being written. Only one part is being summarized at a time.
    """

    def __init__(self, bus: EventBus, settings: ContextSettings) -> None:
        self._bus = bus
        self._settings = settings
        self._items: list[ContextItem] = []
        # Ids handed to the summary and not yet absorbed; None when no summary is running.
        self._summarizing: frozenset[str] | None = None
        # One subscription for every type: the bus keeps order only within a subscription,
        # and an edit must not overtake the phrase it edits.
        bus.subscribe(Event, self._on_event, name="context_window")

    @property
    def items(self) -> tuple[ContextItem, ...]:
        return tuple(self._items)

    async def _on_event(self, event: Event) -> None:
        match event:
            case TranscriptFinal():
                self._add(event.segment)
            case AnswerDone():
                self._add(event.answer)
            case SegmentEdited():
                self._edit(event.segment_id, event.text)
            case SummaryUpdated():
                self._absorb(event.absorbed_ids)
            case SummaryFailed():
                # The part stays in the window and is handed over again on the next phrase.
                self._summarizing = None

    def _add(self, item: ContextItem) -> None:
        # The two channels finish phrases independently, so a long phrase can arrive after
        # a shorter one that started later: placing by start keeps the timeline true.
        bisect.insort(self._items, item, key=_start)
        self._request_summary_if_full()

    def _request_summary_if_full(self) -> None:
        if self._summarizing is not None or not self._items:
            return
        first_start = _start(self._items[0])
        span = max(_end(item) for item in self._items) - first_start
        if span < self._settings.window_seconds:
            return

        cutoff = first_start + self._settings.compress_seconds
        # The part ends at the last phrase that ends before the cutoff, so no phrase is cut.
        # A single phrase longer than the whole part still has to go, or the window would
        # never shrink.
        part = list(takewhile(lambda item: _end(item) <= cutoff, self._items))
        part = part or self._items[:1]
        self._summarizing = frozenset(item.id for item in part)
        log.info("summary_requested", items=len(part), window_seconds=round(span))
        self._bus.publish(SummaryRequested(items=tuple(part)))

    def _edit(self, segment_id: str, text: str) -> None:
        if self._summarizing is not None and segment_id in self._summarizing:
            # The summary is being written from the old text; the edit would be lost the
            # moment the summary absorbs this phrase.
            log.warning("segment_edit_rejected", segment_id=segment_id, reason="summarizing")
            return
        for index, item in enumerate(self._items):
            if item.id == segment_id and isinstance(item, Segment):
                self._items[index] = item.model_copy(update={"text": text})
                return
        log.warning("segment_edit_rejected", segment_id=segment_id, reason="unknown_segment")

    def _absorb(self, absorbed_ids: tuple[str, ...]) -> None:
        absorbed = set(absorbed_ids)
        self._items = [item for item in self._items if item.id not in absorbed]
        self._summarizing = None
