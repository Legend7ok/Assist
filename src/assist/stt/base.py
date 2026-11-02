from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from assist.core.schemas import Speaker

# Both speakers go to one live session as two channels: the free Gladia plan allows only one
# session at a time. The index in this tuple is the channel number.
STEREO_CHANNELS: tuple[Speaker, Speaker] = (Speaker.THEM, Speaker.ME)
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # bytes, int16


class STTError(Exception):
    """Recognition failed; the message says why in words a person can act on."""


@dataclass(frozen=True, slots=True)
class StereoFrame:
    pcm: bytes  # 16 kHz, signed 16-bit little-endian, channels interleaved as STEREO_CHANNELS
    start: float  # meeting time of the first sample

    @property
    def samples(self) -> int:
        """Samples per channel."""
        return len(self.pcm) // (SAMPLE_WIDTH * len(STEREO_CHANNELS))


@dataclass(frozen=True, slots=True)
class Phrase:
    """A phrase as recognized so far, with its times in meeting seconds.

    Partials of one phrase share its id and each replaces the previous one; the final one with
    that id is the last.
    """

    id: str
    speaker: Speaker
    text: str
    start: float
    end: float
    is_final: bool


class Transcriber(Protocol):
    """One live recognition session for both speakers of the meeting.

    Any method raises STTError when the provider refuses the session (wrong key, limit
    reached) or the connection is lost for good.
    """

    async def connect(self) -> None:
        """Open the session; frames can be sent once it returns."""
        ...

    async def send(self, frame: StereoFrame) -> None:
        """Send a frame in which at least one speaker is talking."""
        ...

    def phrases(self) -> AsyncIterator[Phrase]:
        """Yield partials and finals of both speakers as they are recognized.

        Ends after close() has delivered the finals of everything sent.
        """
        ...

    async def close(self) -> None:
        """Stop sending and end the session once the last finals have arrived."""
        ...
