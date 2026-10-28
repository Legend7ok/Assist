"""Device audio to the format speech services take: 16 kHz, mono, int16, fixed-length frames."""

from dataclasses import dataclass

import numpy as np
import soxr

TARGET_RATE = 16_000

_INT16_MAX = np.iinfo(np.int16).max

# A block arriving later than this past its counted time means audio was really missing,
# not a late callback.
_MAX_LATENESS = 0.1


@dataclass(frozen=True, slots=True)
class PcmFrame:
    pcm: bytes  # 16 kHz, mono, signed 16-bit little-endian
    start: float  # time.monotonic() of the first sample


class FrameResampler:
    """Turns blocks from one capture stream into 16 kHz mono int16 frames of `frame_ms`.

    The resampler is streaming and keeps its filter state between blocks, so block borders
    leave no clicks. Samples that do not fill a whole frame wait for the next block.
    """

    def __init__(self, input_rate: int, *, frame_ms: int) -> None:
        self._input_rate = input_rate
        self._frame_samples = TARGET_RATE * frame_ms // 1000
        self._stream = soxr.ResampleStream(input_rate, TARGET_RATE, 1, dtype="float32")
        self._pending = np.empty(0, dtype=np.int16)
        self._next_start: float | None = None

    def process(self, samples: np.ndarray, started_at: float) -> list[PcmFrame]:
        """Take a float32 block shaped (frames, channels) and return the frames it completes."""
        mono = samples.mean(axis=1, dtype=np.float32)
        resampled = self._stream.resample_chunk(mono)
        # Resampling overshoots on loud sharp sounds; unclipped, 1.01 would wrap to -32768.
        pcm = np.rint(np.clip(resampled, -1.0, 1.0) * _INT16_MAX).astype(np.int16)
        pending = np.concatenate((self._pending, pcm))

        # Callbacks come late by a few ms, at stream start in bursts by tens of ms, but never
        # early. So time is counted by samples and pulled back to an earlier arrival, or moved
        # forward only when audio was really missing. Taking every arrival as is would make
        # frame times jitter and even go backwards after a late block.
        block_start = self._next_start
        if (
            block_start is None
            or started_at < block_start
            or started_at - block_start > _MAX_LATENESS
        ):
            block_start = started_at
        block_end = block_start + len(mono) / self._input_rate
        self._next_start = block_end
        # The filter holds back delay() samples: the last sample out is that much older than
        # the end of the block.
        first_start = block_end - (self._stream.delay() + len(pending)) / TARGET_RATE

        size = self._frame_samples
        count = len(pending) // size
        frames = [
            PcmFrame(
                pcm=pending[i * size : (i + 1) * size].tobytes(),
                start=first_start + i * size / TARGET_RATE,
            )
            for i in range(count)
        ]
        self._pending = pending[count * size :]
        return frames
