"""Data shared by every part of the app.

Times are seconds since the session started, on one clock for both audio channels, so phrases
and answers from different sources can be put in one timeline.
"""

from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def new_id() -> str:
    return uuid4().hex


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Speaker(StrEnum):
    ME = "me"
    THEM = "them"


class Segment(Frozen):
    id: str = Field(default_factory=new_id)
    speaker: Speaker
    text: str
    start: float
    end: float


class Answer(Frozen):
    id: str = Field(default_factory=new_id)
    text: str
    at: float


class Message(Frozen):
    role: Literal["system", "user", "assistant"]
    content: str
