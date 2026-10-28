import time

import numpy as np
import pyaudiowpatch as pyaudio
import pytest

from assist.audio.capture import AudioCapture, AudioDevice, AudioDeviceError, DeviceCatalog

SPEAKERS = AudioDevice(1, "Speakers (Realtek)", 0, 48000, is_loopback=False)
HEADPHONES = AudioDevice(2, "Headphones (USB)", 0, 44100, is_loopback=False)
SPEAKERS_LOOPBACK = AudioDevice(3, "Speakers (Realtek) [Loopback]", 2, 48000, is_loopback=True)
HEADPHONES_LOOPBACK = AudioDevice(4, "Headphones (USB) [Loopback]", 2, 44100, is_loopback=True)
MIC = AudioDevice(5, "Microphone (Realtek)", 2, 48000, is_loopback=False)
HEADSET_MIC = AudioDevice(6, "Headset Microphone (USB)", 1, 16000, is_loopback=False)

DEVICES = (SPEAKERS, HEADPHONES, SPEAKERS_LOOPBACK, HEADPHONES_LOOPBACK, MIC, HEADSET_MIC)


def make_catalog(
    devices: tuple[AudioDevice, ...] = DEVICES,
    *,
    default_input: int = MIC.index,
    default_output: int = SPEAKERS.index,
) -> DeviceCatalog:
    return DeviceCatalog(devices, default_input, default_output)


def test_default_loopback_is_the_twin_of_the_default_output():
    catalog = make_catalog(default_output=HEADPHONES.index)

    assert catalog.loopback(None) == HEADPHONES_LOOPBACK


def test_default_loopback_needs_the_exact_twin_name():
    speakers = AudioDevice(1, "Speakers", 0, 48000, is_loopback=False)
    other_twin = AudioDevice(2, "Speakers 2 [Loopback]", 2, 48000, is_loopback=True)
    twin = AudioDevice(3, "Speakers [Loopback]", 2, 48000, is_loopback=True)
    catalog = make_catalog((speakers, other_twin, twin), default_output=speakers.index)

    assert catalog.loopback(None) == twin


def test_default_microphone():
    assert make_catalog().microphone(None) == MIC


def test_loopback_found_by_part_of_the_output_name_in_any_case():
    assert make_catalog().loopback("headphones") == HEADPHONES_LOOPBACK


def test_microphone_name_never_picks_a_loopback_or_output_device():
    # "Realtek" is also in the speakers and their loopback.
    assert make_catalog().microphone("Realtek") == MIC


def test_exact_name_wins_over_partial_matches():
    mic = AudioDevice(1, "Mic", 1, 48000, is_loopback=False)
    mic_2 = AudioDevice(2, "Mic 2", 1, 48000, is_loopback=False)
    catalog = make_catalog((mic, mic_2), default_input=mic.index)

    assert catalog.microphone("mic") == mic


def test_ambiguous_name_is_refused_with_the_matches():
    with pytest.raises(AudioDeviceError) as error:
        make_catalog().microphone("microphone")

    assert MIC.name in str(error.value)
    assert HEADSET_MIC.name in str(error.value)


def test_unknown_name_is_refused_with_the_available_devices():
    with pytest.raises(AudioDeviceError) as error:
        make_catalog().microphone("Webcam")

    message = str(error.value)
    assert "Webcam" in message
    assert MIC.name in message
    assert HEADSET_MIC.name in message
    assert SPEAKERS_LOOPBACK.name not in message


def test_missing_default_output_is_an_error():
    with pytest.raises(AudioDeviceError):
        make_catalog(default_output=-1).loopback(None)


def test_default_output_without_a_loopback_twin_is_an_error():
    with pytest.raises(AudioDeviceError, match="Headphones"):
        make_catalog((HEADPHONES, MIC), default_output=HEADPHONES.index).loopback(None)


def test_missing_default_microphone_is_an_error():
    with pytest.raises(AudioDeviceError):
        make_catalog(default_input=-1).microphone(None)


class FakeStream:
    def __init__(self, callback) -> None:
        self.callback = callback
        self.stopped = False
        self.closed = False
        self.stop_error: OSError | None = None

    def is_active(self) -> bool:
        return not self.stopped

    def stop_stream(self) -> None:
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error

    def close(self) -> None:
        self.closed = True


class FakePyAudio:
    def __init__(self, open_error: OSError | None = None) -> None:
        self.open_error = open_error
        self.open_kwargs: dict = {}
        self.stream: FakeStream | None = None

    def open(self, **kwargs) -> FakeStream:
        if self.open_error is not None:
            raise self.open_error
        self.open_kwargs = kwargs
        self.stream = FakeStream(kwargs["stream_callback"])
        return self.stream


def start_capture(on_audio=lambda samples, started_at: None, device=SPEAKERS_LOOPBACK):
    pa = FakePyAudio()
    capture = AudioCapture(pa, device, frame_ms=20, on_audio=on_audio)
    capture.start()
    assert pa.stream is not None
    return capture, pa, pa.stream


def test_stream_opens_in_the_device_format_with_frame_long_blocks():
    _, pa, _ = start_capture()

    assert pa.open_kwargs["input_device_index"] == SPEAKERS_LOOPBACK.index
    assert pa.open_kwargs["channels"] == 2
    assert pa.open_kwargs["rate"] == 48000
    assert pa.open_kwargs["frames_per_buffer"] == 960
    assert pa.open_kwargs["format"] == pyaudio.paFloat32


def test_block_arrives_as_frames_by_channels_with_its_start_time():
    received: list[tuple[np.ndarray, float]] = []
    _, _, stream = start_capture(lambda samples, started_at: received.append((samples, started_at)))
    left = np.linspace(-1, 1, 960, dtype=np.float32)
    right = np.zeros(960, dtype=np.float32)
    block = np.column_stack([left, right]).tobytes()

    before = time.monotonic()
    result = stream.callback(block, 960, {}, 0)
    after = time.monotonic()

    assert result == (None, pyaudio.paContinue)
    [(samples, started_at)] = received
    assert samples.shape == (960, 2)
    np.testing.assert_array_equal(samples[:, 0], left)
    np.testing.assert_array_equal(samples[:, 1], right)
    # The block ends when it arrives, so it started one block length (20 ms) earlier.
    assert before - 0.02 <= started_at <= after - 0.02


def test_failing_handler_does_not_abort_the_stream():
    calls: list[float] = []

    def broken(samples: np.ndarray, started_at: float) -> None:
        calls.append(started_at)
        raise ValueError("boom")

    _, _, stream = start_capture(broken)
    block = np.zeros((960, 2), dtype=np.float32).tobytes()

    assert stream.callback(block, 960, {}, 0) == (None, pyaudio.paContinue)
    assert stream.callback(block, 960, {}, 0) == (None, pyaudio.paContinue)
    assert len(calls) == 2


def test_stop_closes_the_stream_once():
    capture, _, stream = start_capture()

    capture.stop()
    capture.stop()

    assert stream.stopped
    assert stream.closed
    assert not capture.is_active


def test_stop_closes_the_stream_of_a_vanished_device():
    capture, _, stream = start_capture()
    stream.stop_error = OSError("Unanticipated host error")

    capture.stop()

    assert stream.closed


def test_device_that_cannot_be_opened_is_a_device_error():
    pa = FakePyAudio(open_error=OSError("Invalid sample rate"))
    capture = AudioCapture(pa, MIC, frame_ms=20, on_audio=lambda samples, started_at: None)

    with pytest.raises(AudioDeviceError, match="Microphone"):
        capture.start()


def test_second_start_is_refused():
    capture, _, _ = start_capture()

    with pytest.raises(RuntimeError):
        capture.start()
