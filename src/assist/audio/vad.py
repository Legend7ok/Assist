"""Silence cutting: only speech, with a little audio around it, goes on to recognition."""

from collections import deque
from collections.abc import Callable

import webrtcvad

from assist.audio.resample import TARGET_RATE, PcmFrame

type SpeechDetector = Callable[[bytes], bool]


def webrtc_detector(aggressiveness: int) -> SpeechDetector:
    vad = webrtcvad.Vad(aggressiveness)
    return lambda pcm: vad.is_speech(pcm, TARGET_RATE)


class SpeechFilter:
    """Passes the frames of a phrase and drops the silence between phrases.

    While idle, the last `pre_speech_ms` of frames are held back. A phrase opens once that
    window holds `min_speech_ms` of speech, and the held frames go out first, so the start of
    the first word is kept. The phrase closes after `hangover_ms` of silence, which is still
    passed on: recognition needs it to see that the phrase ended.
    """

    def __init__(
        self,
        is_speech: SpeechDetector,
        *,
        frame_ms: int,
        pre_speech_ms: int,
        min_speech_ms: int,
        hangover_ms: int,
    ) -> None:
        self._is_speech = is_speech
        self._min_speech_frames = -(-min_speech_ms // frame_ms)  # rounded up
        self._hangover_frames = hangover_ms // frame_ms
        # The window has to fit the speech needed to open a phrase, or it would never open.
        window = max(pre_speech_ms // frame_ms, self._min_speech_frames)
        self._held: deque[tuple[PcmFrame, bool]] = deque(maxlen=window)
        self._in_phrase = False
        self._silent_frames = 0

    @property
    def in_phrase(self) -> bool:
        return self._in_phrase

    def process(self, frame: PcmFrame) -> list[PcmFrame]:
        """Return the frames to pass on: none, this one, or the held ones and this one."""
        speech = self._is_speech(frame.pcm)

        if self._in_phrase:
            self._silent_frames = 0 if speech else self._silent_frames + 1
            if self._silent_frames <= self._hangover_frames:
                return [frame]
            # Silence past the hangover: the phrase is over, and this frame is the first one
            # held for the next phrase.
            self._in_phrase = False

        self._held.append((frame, speech))
        if sum(is_speech for _, is_speech in self._held) < self._min_speech_frames:
            return []
        frames = [held for held, _ in self._held]
        self._held.clear()
        self._in_phrase = True
        self._silent_frames = 0
        return frames
