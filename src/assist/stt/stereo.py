import numpy as np

from assist.core.events import AudioFrame
from assist.core.schemas import Speaker
from assist.stt.base import STEREO_CHANNELS, StereoFrame

# Covers how far one device's frames can trail the other's on the way to the bus.
DEFAULT_BUFFER_MS = 100


class StereoAssembler:
    """Joins the speech frames of both speakers into stereo frames on one meeting time grid.

    The grid step is the frame length, and a frame takes the slot nearest its start. A slot
    goes out once it is `buffer_ms` older than the newest frame of either speaker, so a
    device whose frames reach the bus a little later still meets the other in the same slot.
    A silent speaker's channel is filled with silence; slots where nobody speaks are not sent.

    A frame arriving for a slot already sent takes the nearest free slot instead of being
    lost: a few milliseconds off in time cost recognition nothing, a gap in a word does.
    """

    def __init__(self, *, frame_ms: int, buffer_ms: int = DEFAULT_BUFFER_MS) -> None:
        self._frame_seconds = frame_ms / 1000
        self._buffer_slots = buffer_ms // frame_ms
        self._pending: dict[int, list[bytes | None]] = {}
        # Slots before this one are sent; None until the first one is.
        self._sent_until: int | None = None
        self._newest: int | None = None
        # The slot each speaker's next frame takes while the speaker keeps talking.
        self._next: dict[Speaker, int] = {}

    def add(self, frame: AudioFrame) -> list[StereoFrame]:
        """Place a frame and return the stereo frames that are complete now, in time order."""
        slot = self._place(frame)
        channels = self._pending.setdefault(slot, [None] * len(STEREO_CHANNELS))
        channels[STEREO_CHANNELS.index(frame.speaker)] = frame.pcm
        self._next[frame.speaker] = slot + 1
        self._newest = slot if self._newest is None else max(self._newest, slot)
        return self._send_until(self._newest - self._buffer_slots + 1)

    def flush(self) -> list[StereoFrame]:
        """Return every frame still held: nobody is talking now, or recognition stops."""
        if self._newest is None:
            return []
        return self._send_until(self._newest + 1)

    def _place(self, frame: AudioFrame) -> int:
        slot = round(frame.start / self._frame_seconds)
        expected = self._next.get(frame.speaker)
        # Within a phrase a speaker's frames follow each other with no gap, but device clock
        # drift moves their starts a little against the grid. Rounded alone, two frames
        # could land in one slot or leave a slot empty, so the next slot is kept unless the
        # frame starts clearly later: a new phrase after a pause.
        if expected is not None and slot <= expected + 1:
            slot = expected
        if self._sent_until is not None:
            slot = max(slot, self._sent_until)
        channel = STEREO_CHANNELS.index(frame.speaker)
        while (taken := self._pending.get(slot)) is not None and taken[channel] is not None:
            slot += 1
        return slot

    def _send_until(self, end: int) -> list[StereoFrame]:
        ready = sorted(slot for slot in self._pending if slot < end)
        if self._sent_until is None or end > self._sent_until:
            self._sent_until = end
        return [self._stereo(slot, self._pending.pop(slot)) for slot in ready]

    def _stereo(self, slot: int, channels: list[bytes | None]) -> StereoFrame:
        size = next(len(pcm) for pcm in channels if pcm is not None)
        silence = bytes(size)
        samples = [np.frombuffer(pcm or silence, dtype=np.int16) for pcm in channels]
        return StereoFrame(pcm=np.column_stack(samples).tobytes(), start=slot * self._frame_seconds)
