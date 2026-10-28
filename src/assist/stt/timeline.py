import bisect
from dataclasses import dataclass

from assist.stt.base import SAMPLE_RATE, StereoFrame

# Frames closer than this in meeting time are one continuous piece of speech: their starts
# are float sums and never meet exactly.
_CONTIGUOUS = 0.001


@dataclass(slots=True)
class _Piece:
    sent_start: float  # seconds of audio sent before this piece
    meeting_start: float
    samples: int

    @property
    def duration(self) -> float:
        return self.samples / SAMPLE_RATE


class Timeline:
    """Turns times in the audio sent to recognition into meeting times.

    Recognition counts time by the audio it got, and the silence between phrases is never
    sent, so its times fall behind the meeting by every pause cut. The timeline remembers
    where each continuous piece of sent speech lies in the meeting. One timeline serves one
    session: a new session counts its audio from zero again.
    """

    def __init__(self) -> None:
        self._pieces: list[_Piece] = []
        # Counted in samples: a sum of float durations would drift over a long meeting.
        self._sent_samples = 0

    @property
    def sent_seconds(self) -> float:
        return self._sent_samples / SAMPLE_RATE

    def add(self, frame: StereoFrame) -> None:
        """Record a frame in the order it is sent."""
        last = self._pieces[-1] if self._pieces else None
        if last is not None and abs(last.meeting_start + last.duration - frame.start) < _CONTIGUOUS:
            last.samples += frame.samples
        else:
            self._pieces.append(
                _Piece(self._sent_samples / SAMPLE_RATE, frame.start, frame.samples)
            )
        self._sent_samples += frame.samples

    def span(self, start: float, end: float) -> tuple[float, float]:
        """Meeting times of a phrase given by its start and end in the sent audio.

        A time on the border of two pieces is both the end of one and the start of the next,
        which can be minutes apart in the meeting: a phrase starting there starts in the
        later piece, a phrase ending there ends in the earlier one.
        """
        if not self._pieces:
            raise ValueError("No audio has been sent yet")
        return self._at(start, later=True), self._at(end, later=False)

    def _at(self, sent: float, *, later: bool) -> float:
        search = bisect.bisect_right if later else bisect.bisect_left
        index = max(search(self._pieces, sent, key=lambda piece: piece.sent_start) - 1, 0)
        piece = self._pieces[index]
        # Recognition rounds its times and can step a little outside the audio it got;
        # nothing was said before the first sample or after the last one.
        offset = min(max(sent - piece.sent_start, 0.0), piece.duration)
        return piece.meeting_start + offset
