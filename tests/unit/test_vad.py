from assist.audio.resample import PcmFrame
from assist.audio.vad import SpeechFilter, webrtc_detector

SPEECH = b"S"
SILENCE = b"."
FRAME_MS = 20


def make_filter(
    *, pre_speech_ms: int = 100, min_speech_ms: int = 40, hangover_ms: int = 60
) -> SpeechFilter:
    return SpeechFilter(
        lambda pcm: pcm == SPEECH,
        frame_ms=FRAME_MS,
        pre_speech_ms=pre_speech_ms,
        min_speech_ms=min_speech_ms,
        hangover_ms=hangover_ms,
    )


def run(speech_filter: SpeechFilter, pattern: str) -> list[int]:
    """Feed one frame per character ("S" speech, "." silence); return numbers of passed frames."""
    passed: list[int] = []
    for n, mark in enumerate(pattern):
        pcm = SPEECH if mark == "S" else SILENCE
        passed += [int(f.start) for f in speech_filter.process(PcmFrame(pcm=pcm, start=n))]
    return passed


def test_silence_is_dropped():
    assert run(make_filter(), "." * 50) == []


def test_phrase_starts_with_the_held_audio_before_it():
    # 100 ms window = 5 frames; the phrase opens on the second speech frame (40 ms).
    speech_filter = make_filter()

    assert run(speech_filter, "..........SS") == [7, 8, 9, 10, 11]
    assert speech_filter.in_phrase


def test_lone_click_does_not_open_a_phrase():
    assert run(make_filter(), "....S........") == []


def test_speech_spread_over_the_window_opens_a_phrase():
    assert run(make_filter(), "....S.S") == [2, 3, 4, 5, 6]


def test_silence_after_speech_is_passed_until_the_hangover_ends():
    # 60 ms hangover = 3 silent frames after the phrase, then nothing.
    speech_filter = make_filter()

    assert run(speech_filter, "SS......") == [0, 1, 2, 3, 4]
    assert not speech_filter.in_phrase


def test_pause_shorter_than_the_hangover_stays_inside_the_phrase():
    speech_filter = make_filter()

    assert run(speech_filter, "SS..SS..S") == list(range(9))
    assert speech_filter.in_phrase


def test_next_phrase_does_not_repeat_frames_already_passed():
    passed = run(make_filter(), "SS......SS....SS..")

    assert passed == sorted(set(passed))
    assert passed == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


def test_without_hangover_the_first_silent_frame_closes_the_phrase():
    speech_filter = make_filter(hangover_ms=0)

    assert run(speech_filter, "SSSS..") == [0, 1, 2, 3]
    assert not speech_filter.in_phrase


def test_window_fits_the_speech_needed_to_open_even_without_pre_buffer():
    assert run(make_filter(pre_speech_ms=0, min_speech_ms=40), "...SS") == [3, 4]


def test_webrtc_detector_takes_a_frame_and_hears_no_speech_in_silence():
    is_speech = webrtc_detector(2)

    assert is_speech(bytes(640)) is False
