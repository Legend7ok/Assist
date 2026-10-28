import pytest

from assist.stt.base import SAMPLE_RATE, StereoFrame
from assist.stt.timeline import Timeline

FRAME = 0.02


def speech(timeline: Timeline, start: float, seconds: float) -> None:
    """Send continuous speech starting at a meeting time, in 20 ms stereo frames."""
    samples = round(FRAME * SAMPLE_RATE)
    for n in range(round(seconds / FRAME)):
        # Starts as float sums, the way the stereo frames get them.
        timeline.add(StereoFrame(pcm=bytes(samples * 4), start=start + n * FRAME))


@pytest.fixture
def cut_pause() -> Timeline:
    """1 s of speech at 10 s of the meeting, a 29 s pause not sent, 1 s of speech at 40 s."""
    timeline = Timeline()
    speech(timeline, 10.0, 1.0)
    speech(timeline, 40.0, 1.0)
    return timeline


def test_continuous_speech_keeps_its_times():
    timeline = Timeline()
    speech(timeline, 10.0, 5.0)

    assert timeline.span(0.2, 4.7) == pytest.approx((10.2, 14.7))


def test_phrase_after_a_cut_pause_gets_its_meeting_time(cut_pause):
    assert cut_pause.span(0.5, 0.9) == pytest.approx((10.5, 10.9))
    assert cut_pause.span(1.2, 1.5) == pytest.approx((40.2, 40.5))


def test_phrase_across_a_cut_pause_spans_it(cut_pause):
    assert cut_pause.span(0.5, 1.5) == pytest.approx((10.5, 40.5))


def test_phrase_starting_on_the_border_starts_in_the_later_piece(cut_pause):
    assert cut_pause.span(1.0, 1.4) == pytest.approx((40.0, 40.4))


def test_phrase_ending_on_the_border_ends_in_the_earlier_piece(cut_pause):
    assert cut_pause.span(0.6, 1.0) == pytest.approx((10.6, 11.0))


def test_times_outside_the_sent_audio_stop_at_its_edges(cut_pause):
    assert cut_pause.span(-0.05, 0.3) == pytest.approx((10.0, 10.3))
    assert cut_pause.span(1.8, 2.3) == pytest.approx((40.8, 41.0))


def test_frame_sent_out_of_meeting_order_keeps_its_own_time():
    timeline = Timeline()
    speech(timeline, 10.0, 0.1)
    speech(timeline, 9.98, FRAME)
    speech(timeline, 10.1, 0.1)

    assert timeline.span(0.1, 0.12) == pytest.approx((9.98, 10.0))
    assert timeline.span(0.12, 0.22) == pytest.approx((10.1, 10.2))


def test_nothing_sent_yet_is_an_error():
    with pytest.raises(ValueError, match="No audio"):
        Timeline().span(0.0, 1.0)
