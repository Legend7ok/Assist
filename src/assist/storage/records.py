"""Records of a session file: one JSON object per line, told apart by `type`.

Records have their own types instead of dumped events: the file outlives the code that wrote
it, so renaming an event must not make older sessions unreadable.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, JsonValue, TypeAdapter

from assist.core.events import Component
from assist.core.schemas import Answer, Frozen, Segment


class SessionStarted(Frozen):
    type: Literal["session_started"] = "session_started"
    started_at: AwareDatetime
    info: dict[str, JsonValue] = Field(default_factory=dict)


class PhraseRecord(Frozen):
    type: Literal["phrase"] = "phrase"
    segment: Segment


class PhraseEditedRecord(Frozen):
    type: Literal["phrase_edited"] = "phrase_edited"
    at: float
    segment_id: str
    text: str


class AnswerRecord(Frozen):
    type: Literal["answer"] = "answer"
    answer: Answer


class AnswerFailedRecord(Frozen):
    type: Literal["answer_failed"] = "answer_failed"
    at: float
    answer_id: str
    reason: str


class SummaryRecord(Frozen):
    type: Literal["summary"] = "summary"
    at: float
    text: str
    absorbed_ids: tuple[str, ...]


class SummaryEditedRecord(Frozen):
    type: Literal["summary_edited"] = "summary_edited"
    at: float
    text: str


class NotesRecord(Frozen):
    type: Literal["notes"] = "notes"
    at: float
    text: str


class StatusErrorRecord(Frozen):
    type: Literal["status_error"] = "status_error"
    at: float
    component: Component
    detail: str


class SessionEnded(Frozen):
    type: Literal["session_ended"] = "session_ended"
    at: float


Record = Annotated[
    SessionStarted
    | PhraseRecord
    | PhraseEditedRecord
    | AnswerRecord
    | AnswerFailedRecord
    | SummaryRecord
    | SummaryEditedRecord
    | NotesRecord
    | StatusErrorRecord
    | SessionEnded,
    Field(discriminator="type"),
]
record_adapter: TypeAdapter[Record] = TypeAdapter(Record)
