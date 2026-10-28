"""Events the parts of the app exchange through the bus.

Each part publishes and subscribes to these types only and never calls another part directly.
"""

from enum import StrEnum

from assist.core.schemas import Answer, Frozen, Segment, Speaker


class Event(Frozen):
    pass


class Component(StrEnum):
    AUDIO = "audio"
    STT = "stt"
    LLM_ANSWER = "llm_answer"
    LLM_SUMMARY = "llm_summary"


class ComponentState(StrEnum):
    OK = "ok"
    CONNECTING = "connecting"
    BUSY = "busy"
    ERROR = "error"


class AudioFrame(Event):
    speaker: Speaker
    pcm: bytes  # 16 kHz, mono, signed 16-bit little-endian


class TranscriptPartial(Event):
    """The phrase being spoken right now; replaced by the next partial or by the final."""

    speaker: Speaker
    text: str


class TranscriptFinal(Event):
    segment: Segment


class AnswerRequested(Event):
    """AltGr or the button: answer now. A newer request cancels the answer being written."""

    # No request text here: AltGr is caught by a keyboard hook that cannot see the window's
    # field. The current request comes from RequestChanged instead.


class AnswerDelta(Event):
    answer_id: str
    text: str


class AnswerDone(Event):
    answer: Answer


class AnswerFailed(Event):
    answer_id: str
    reason: str


class NotesChanged(Event):
    text: str


class RequestChanged(Event):
    """The one-off request field changed; empty text after an answer has used the request."""

    text: str


class SummaryRequested(Event):
    """The oldest part of the window goes into the summary.

    Until SummaryUpdated or SummaryFailed these items stay in the context and cannot be edited.
    """

    items: tuple[Segment | Answer, ...]


class SummaryFailed(Event):
    """The requested part was not summarized; it stays in the window for the next attempt."""

    reason: str


class SummaryUpdated(Event):
    """The summary absorbed older phrases and answers; their ids leave the window and the feed."""

    text: str
    absorbed_ids: tuple[str, ...]


class SegmentEdited(Event):
    segment_id: str
    text: str


class SummaryEdited(Event):
    text: str


class StatusChanged(Event):
    component: Component
    state: ComponentState
    detail: str = ""
