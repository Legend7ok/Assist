import numpy as np
import pytest

from assist.core.events import AudioFrame
from assist.core.schemas import Speaker
from assist.stt.base import StereoFrame
from assist.stt.stereo import StereoAssembler

FRAME = 0.02
SAMPLES = 320  # 20 ms at 16 kHz


def frame(speaker: Speaker, start: float, value: int) -> AudioFrame:
    """A frame whose every sample is `value`, so it can be found in the stereo output."""
    pcm = np.full(SAMPLES, value, dtype=np.int16).tobytes()
    return AudioFrame(speaker=speaker, pcm=pcm, start=start)


def run(*frames: AudioFrame) -> list[StereoFrame]:
    """Feed frames in arrival order and flush at the end, like the end of speech."""
    assembler = StereoAssembler(frame_ms=20)
    out: list[StereoFrame] = []
    for f in frames:
        out += assembler.add(f)
    return out + assembler.flush()


def channels(stereo: StereoFrame) -> tuple[int, int]:
    """The value of each channel: THEM, ME; 0 is silence."""
    samples = np.frombuffer(stereo.pcm, dtype=np.int16).reshape(-1, 2)
    them, me = samples[0]
    assert (samples == samples[0]).all()
    return int(them), int(me)


def talk(speaker: Speaker, start: float, count: int, *, first_value: int = 1, step=FRAME):
    return [frame(speaker, start + n * step, first_value + n) for n in range(count)]


def test_speech_of_one_speaker_goes_with_silence_in_the_other_channel():
    out = run(*talk(Speaker.THEM, 1.0, 10))

    assert [channels(s) for s in out] == [(n, 0) for n in range(1, 11)]
    assert [s.start for s in out] == pytest.approx([1.0 + n * FRAME for n in range(10)])
    assert all(len(s.pcm) == SAMPLES * 4 for s in out)


def test_simultaneous_speech_shares_the_frames():
    them = talk(Speaker.THEM, 1.0, 10)
    # The other device captures a few milliseconds off the grid.
    me = talk(Speaker.ME, 1.007, 10, first_value=101)

    out = run(*[f for pair in zip(them, me, strict=True) for f in pair])

    assert [channels(s) for s in out] == [(n, 100 + n) for n in range(1, 11)]


def test_overlapping_speech_is_joined_where_it_overlaps():
    them = talk(Speaker.THEM, 1.0, 10)
    me = talk(Speaker.ME, 1.1, 10, first_value=101)

    out = run(*sorted([*them, *me], key=lambda f: f.start))

    assert [channels(s) for s in out] == [
        *[(n, 0) for n in range(1, 6)],
        *[(n, 95 + n) for n in range(6, 11)],
        *[(0, n) for n in range(106, 111)],
    ]
    assert out[0].start == pytest.approx(1.0)
    assert out[-1].start == pytest.approx(1.28)


def test_frames_of_the_other_device_within_the_buffer_meet_in_the_same_slot():
    them = talk(Speaker.THEM, 1.0, 10)
    me = talk(Speaker.ME, 1.0, 10, first_value=101)

    # The microphone's frames reach the bus 80 ms behind the loopback's.
    out = run(*them[:4], *[f for pair in zip(them[4:], me, strict=False) for f in pair], *me[6:])

    assert [channels(s) for s in out] == [(n, 100 + n) for n in range(1, 11)]


def test_frame_later_than_the_buffer_takes_the_nearest_free_slot():
    them = talk(Speaker.THEM, 1.0, 20)
    late = frame(Speaker.ME, 1.0, 101)

    out = run(*them, late)

    found = [(s.start, channels(s)) for s in out if channels(s)[1] == 101]
    assert len(found) == 1
    start, (them_value, _) = found[0]
    # Slots up to 1.28 are already sent when it arrives; the first one still held is 1.30.
    assert start == pytest.approx(1.3)
    assert them_value == 16
    assert len(out) == 20


def test_silence_between_phrases_is_not_sent():
    out = run(*talk(Speaker.THEM, 1.0, 3), *talk(Speaker.THEM, 5.0, 3, first_value=4))

    assert [s.start for s in out] == pytest.approx([1.0, 1.02, 1.04, 5.0, 5.02, 5.04])


def test_clock_drift_between_devices_loses_and_repeats_no_frame():
    # One device runs 0.1% slow, the other 0.1% fast, through a minute of talking at once.
    them = talk(Speaker.THEM, 1.0, 3000, step=FRAME * 1.001)
    me = talk(Speaker.ME, 1.0, 3000, first_value=10_001, step=FRAME * 0.999)

    out = run(*sorted([*them, *me], key=lambda f: f.start))

    values = [channels(s) for s in out]
    assert [them for them, _ in values if them] == list(range(1, 3001))
    assert [me for _, me in values if me] == list(range(10_001, 13_001))
    starts = [s.start for s in out]
    assert starts == sorted(starts)


def test_flush_with_nothing_held_returns_nothing():
    assert StereoAssembler(frame_ms=20).flush() == []
