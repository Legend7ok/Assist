import numpy as np
import pytest

from assist.audio.resample import TARGET_RATE, FrameResampler, PcmFrame

FRAME_MS = 20
FRAME_SAMPLES = TARGET_RATE * FRAME_MS // 1000
# The filter starts from silence: its first outputs ramp up and are not the signal yet.
WARM_UP = FRAME_SAMPLES * 5


def tone(rate: int, seconds: float, *, hz: float = 1000, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def stereo(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    return np.column_stack((left, left if right is None else right))


def feed(
    signal: np.ndarray, rate: int, block_ms: int = FRAME_MS, start: float = 100.0
) -> list[PcmFrame]:
    """Push a (frames, channels) signal through in device-sized blocks, like the capture does."""
    resampler = FrameResampler(rate, frame_ms=FRAME_MS)
    block = rate * block_ms // 1000
    frames: list[PcmFrame] = []
    for offset in range(0, len(signal), block):
        frames += resampler.process(signal[offset : offset + block], start + offset / rate)
    return frames


def samples_of(frames: list[PcmFrame]) -> np.ndarray:
    return np.frombuffer(b"".join(f.pcm for f in frames), dtype=np.int16)


@pytest.mark.parametrize("rate", [48_000, 44_100, 16_000])
def test_every_frame_is_20_ms_of_16_khz_and_the_length_is_kept(rate):
    frames = feed(stereo(tone(rate, 1.0)), rate)

    assert all(len(f.pcm) == FRAME_SAMPLES * 2 for f in frames)
    # One second in gives one second out, short of what the filter still holds.
    assert TARGET_RATE - 2 * FRAME_SAMPLES <= len(samples_of(frames)) <= TARGET_RATE


def test_tone_keeps_its_frequency():
    rate = 48_000
    samples = samples_of(feed(stereo(tone(rate, 1.0, hz=1000)), rate))[WARM_UP:]

    spectrum = np.abs(np.fft.rfft(samples))
    peak_hz = np.argmax(spectrum) * TARGET_RATE / len(samples)

    assert peak_hz == pytest.approx(1000, abs=5)


def test_block_borders_leave_no_jumps():
    rate = 44_100  # 882 frames per block: borders never fall on whole output samples
    hz, amplitude = 1000, 0.5
    samples = samples_of(feed(stereo(tone(rate, 1.0, hz=hz, amplitude=amplitude)), rate))
    steps = np.abs(np.diff(samples[WARM_UP:].astype(np.int32)))

    # A sine cannot change faster than its slope; a click at a border would.
    max_step = amplitude * 32767 * 2 * np.pi * hz / TARGET_RATE
    assert steps.max() <= max_step * 1.05


def test_streaming_matches_resampling_the_whole_signal_at_once():
    rate = 48_000
    signal = stereo(tone(rate, 0.5))
    whole = samples_of(feed(signal, rate, block_ms=500))
    in_blocks = samples_of(feed(signal, rate, block_ms=20))

    assert np.abs(whole.astype(np.int32) - in_blocks[: len(whole)]).max() <= 1


def test_channels_are_mixed_to_mono_by_average():
    rate = 48_000
    length = rate // 2
    left = np.full(length, 0.4, dtype=np.float32)
    right = np.full(length, 0.2, dtype=np.float32)

    samples = samples_of(feed(stereo(left, right), rate))[WARM_UP:]

    np.testing.assert_allclose(samples, 0.3 * 32767, atol=2)


def test_full_scale_does_not_wrap_around():
    rate = 48_000
    loud = np.ones(rate // 2, dtype=np.float32)

    samples = samples_of(feed(stereo(loud), rate))

    assert samples.min() >= 0
    assert samples[WARM_UP:].min() == 32767


def test_frames_follow_each_other_in_time():
    rate = 44_100
    frames = feed(stereo(tone(rate, 1.0)), rate, start=100.0)
    starts = np.array([f.start for f in frames])

    np.testing.assert_allclose(np.diff(starts), FRAME_MS / 1000, atol=1e-3)
    # The first sample out is the first sample in, delayed by nothing but the filter.
    assert starts[0] == pytest.approx(100.0, abs=0.005)


def run_blocks(arrivals: list[float]) -> list[PcmFrame]:
    rate = 48_000
    block = stereo(tone(rate, FRAME_MS / 1000))
    resampler = FrameResampler(rate, frame_ms=FRAME_MS)
    frames: list[PcmFrame] = []
    for started_at in arrivals:
        frames += resampler.process(block, started_at)
    return frames


STEADY = [100.0 + n * 0.02 for n in range(20)]


def test_late_callbacks_do_not_move_frame_times():
    jittery = [t + (0.05 if n == 10 else 0.004 * (n % 3)) for n, t in enumerate(STEADY)]

    steady, late = run_blocks(STEADY), run_blocks(jittery)

    assert [f.start for f in late] == pytest.approx([f.start for f in steady], abs=1e-9)


def test_earlier_arrival_pulls_frame_times_back():
    late_start = [t + 0.03 for t in STEADY[:10]] + STEADY[10:]

    steady, corrected = run_blocks(STEADY), run_blocks(late_start)

    assert corrected[-1].start == pytest.approx(steady[-1].start, abs=1e-9)


def test_frame_time_follows_the_capture_clock_after_lost_audio():
    # 200 ms of blocks never arrived: what follows keeps its real time.
    lossy = STEADY[:10] + [t + 0.2 for t in STEADY[10:]]

    steady, after_loss = run_blocks(STEADY), run_blocks(lossy)

    assert after_loss[-1].start - steady[-1].start == pytest.approx(0.2, abs=1e-9)
