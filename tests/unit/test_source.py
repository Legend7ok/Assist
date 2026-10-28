import asyncio
import threading

import numpy as np
import pytest

from assist.audio.capture import AudioCallback, AudioDevice, AudioDeviceError
from assist.audio.source import AudioSource
from assist.core.bus import EventBus
from assist.core.events import AudioFrame, Component, ComponentState, Event, StatusChanged
from assist.core.schemas import Speaker
from assist.settings import AudioSettings

RATE = 48_000
BLOCK = RATE // 50  # 20 ms
SESSION_START = 1000.0

LOOPBACK = AudioDevice(1, "Headphones [Loopback]", 2, RATE, is_loopback=True)
MIC = AudioDevice(2, "Microphone", 2, RATE, is_loopback=False)


class FakeCapture:
    def __init__(self, on_audio: AudioCallback) -> None:
        self.on_audio = on_audio
        self.started = False
        self.stopped = False
        self.is_active = False

    def start(self) -> None:
        self.started = self.is_active = True

    def stop(self) -> None:
        self.stopped = True
        self.is_active = False


class FakeBackend:
    def __init__(self, *, failing: Speaker | None = None) -> None:
        self.failing = failing
        self.captures: dict[Speaker, FakeCapture] = {}
        self.closed = False

    def device(self, speaker: Speaker) -> AudioDevice:
        if speaker is self.failing:
            raise AudioDeviceError(f"No device for {speaker}")
        return LOOPBACK if speaker is Speaker.THEM else MIC

    def open(self, device: AudioDevice, on_audio: AudioCallback) -> FakeCapture:
        capture = FakeCapture(on_audio)
        self.captures[Speaker.THEM if device is LOOPBACK else Speaker.ME] = capture
        return capture

    def close(self) -> None:
        self.closed = True


def loud(pcm: bytes) -> bool:
    return np.abs(np.frombuffer(pcm, dtype=np.int16)).mean() > 1000


class Recorder:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def __call__(self, event: Event) -> None:
        self.events.append(event)

    def frames(self, speaker: Speaker | None = None) -> list[AudioFrame]:
        return [
            e
            for e in self.events
            if isinstance(e, AudioFrame) and (speaker is None or e.speaker is speaker)
        ]

    def statuses(self) -> list[StatusChanged]:
        return [e for e in self.events if isinstance(e, StatusChanged)]


@pytest.fixture
async def bus():
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.close()


@pytest.fixture
def recorder(bus: EventBus) -> Recorder:
    recorder = Recorder()
    bus.subscribe(Event, recorder)
    return recorder


def make_source(bus: EventBus, backend: FakeBackend, *, watch_interval: float = 0.5):
    return AudioSource(
        bus,
        AudioSettings(),
        session_start=SESSION_START,
        backend=backend,
        detector_factory=lambda: loud,
        watch_interval=watch_interval,
    )


def blocks(pattern: str) -> list[np.ndarray]:
    """One 20 ms stereo block per character: "S" a loud tone, "." silence."""
    t = np.arange(BLOCK) / RATE
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    silence = np.zeros(BLOCK, dtype=np.float32)
    return [np.column_stack([tone if m == "S" else silence] * 2) for m in pattern]


def play(capture: FakeCapture, pattern: str, *, start: float = SESSION_START) -> None:
    """Deliver blocks from another thread, the way PortAudio calls back."""

    def run() -> None:
        for n, block in enumerate(blocks(pattern)):
            capture.on_audio(block, start + n * BLOCK / RATE)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()


async def settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def test_start_reports_audio_ok_and_starts_both_channels(bus, recorder):
    backend = FakeBackend()
    source = make_source(bus, backend)

    await source.start()
    await settle()

    assert backend.captures[Speaker.THEM].started
    assert backend.captures[Speaker.ME].started
    assert recorder.statuses() == [
        StatusChanged(component=Component.AUDIO, state=ComponentState.OK)
    ]
    await source.stop()


async def test_speech_reaches_the_bus_with_its_speaker_and_silence_does_not(bus, recorder):
    backend = FakeBackend()
    source = make_source(bus, backend)
    await source.start()

    play(backend.captures[Speaker.THEM], "." * 50 + "S" * 25 + "." * 50)
    play(backend.captures[Speaker.ME], "." * 125)
    await settle()

    them = recorder.frames(Speaker.THEM)
    assert recorder.frames(Speaker.ME) == []
    # 0.5 s of speech plus the held audio before it and the hangover after it, not 2.5 s.
    assert 25 <= len(them) <= 25 + (300 + 600) // 20 + 2
    assert all(len(f.pcm) == 640 for f in them)
    await source.stop()


async def test_frame_time_is_counted_from_the_session_start(bus, recorder):
    backend = FakeBackend()
    source = make_source(bus, backend)
    await source.start()

    # Speech starts 3 s into the session.
    play(backend.captures[Speaker.ME], "." * 20 + "S" * 25, start=SESSION_START + 2.6)
    await settle()

    starts = [f.start for f in recorder.frames(Speaker.ME)]
    # The 300 ms held back end with the 60 ms of speech that opened the phrase: 240 ms of the
    # audio before the speech go out first.
    assert starts[0] == pytest.approx(3.0 - 0.24, abs=0.03)
    assert starts == sorted(starts)
    await source.stop()


async def test_stop_closes_both_channels_and_the_devices(bus, recorder):
    backend = FakeBackend()
    source = make_source(bus, backend)
    await source.start()

    await source.stop()

    assert backend.captures[Speaker.THEM].stopped
    assert backend.captures[Speaker.ME].stopped
    assert backend.closed


async def test_device_error_is_reported_and_nothing_stays_open(bus, recorder):
    backend = FakeBackend(failing=Speaker.ME)
    source = make_source(bus, backend)

    with pytest.raises(AudioDeviceError):
        await source.start()
    await settle()

    assert backend.captures[Speaker.THEM].stopped
    assert backend.closed
    [status] = recorder.statuses()
    assert status.state is ComponentState.ERROR
    assert "me" in status.detail


async def test_device_that_stops_is_reported_once(bus, recorder):
    backend = FakeBackend()
    source = make_source(bus, backend, watch_interval=0.01)
    await source.start()

    backend.captures[Speaker.ME].is_active = False
    await asyncio.sleep(0.1)
    await settle()

    errors = [s for s in recorder.statuses() if s.state is ComponentState.ERROR]
    assert len(errors) == 1
    assert MIC.name in errors[0].detail
    await source.stop()


async def test_second_start_is_refused(bus, recorder):
    source = make_source(bus, FakeBackend())
    await source.start()

    with pytest.raises(RuntimeError):
        await source.start()
    await source.stop()
