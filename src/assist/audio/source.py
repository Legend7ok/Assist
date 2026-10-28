"""Both audio channels from the devices to the bus, as frames of speech."""

import asyncio
from collections.abc import Callable
from typing import Protocol

import numpy as np
import structlog

from assist.audio.capture import AudioCallback, AudioDevice, AudioDeviceError, WasapiBackend
from assist.audio.resample import FrameResampler
from assist.audio.vad import SpeechDetector, SpeechFilter, webrtc_detector
from assist.core.bus import EventBus
from assist.core.events import AudioFrame, Component, ComponentState, Event, StatusChanged
from assist.core.schemas import Speaker
from assist.settings import AudioSettings

log = structlog.get_logger(__name__)

# The loopback first: without the meeting audio there is nothing to answer, so its error is
# the one to report when both devices fail.
_CHANNELS = (Speaker.THEM, Speaker.ME)
_KIND = {Speaker.THEM: "loopback", Speaker.ME: "microphone"}


class Capture(Protocol):
    @property
    def is_active(self) -> bool: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


class CaptureBackend(Protocol):
    def device(self, speaker: Speaker) -> AudioDevice: ...

    def open(self, device: AudioDevice, on_audio: AudioCallback) -> Capture: ...

    def close(self) -> None: ...


class _Channel:
    """Capture → 16 kHz mono frames → speech only, for one speaker, on its PortAudio thread."""

    def __init__(
        self,
        speaker: Speaker,
        device: AudioDevice,
        settings: AudioSettings,
        detector: SpeechDetector,
        *,
        session_start: float,
        publish: Callable[[Event], None],
    ) -> None:
        self.speaker = speaker
        self.device = device
        self._session_start = session_start
        self._publish = publish
        self._resampler = FrameResampler(device.sample_rate, frame_ms=settings.frame_ms)
        self._filter = SpeechFilter(
            detector,
            frame_ms=settings.frame_ms,
            pre_speech_ms=settings.pre_speech_ms,
            min_speech_ms=settings.min_speech_ms,
            hangover_ms=settings.hangover_ms,
        )

    def on_audio(self, samples: np.ndarray, started_at: float) -> None:
        for frame in self._resampler.process(samples, started_at):
            was_in_phrase = self._filter.in_phrase
            for speech in self._filter.process(frame):
                self._publish(
                    AudioFrame(
                        speaker=self.speaker,
                        pcm=speech.pcm,
                        start=speech.start - self._session_start,
                    )
                )
            if self._filter.in_phrase != was_in_phrase:
                at = round(frame.start - self._session_start, 3)
                name = "speech_started" if self._filter.in_phrase else "speech_ended"
                log.debug(name, channel=self.speaker, at=at)


class AudioSource:
    """Captures the meeting (loopback, THEM) and the user (microphone, ME) and publishes the
    speech in both as AudioFrame events; silence between phrases is not published.

    Each channel works on its own PortAudio thread and publishes with publish_threadsafe.
    StatusChanged for AUDIO tells when capture works and when a device failed to open or
    stopped. A stopped device is not reopened: the source has to be restarted.
    """

    def __init__(
        self,
        bus: EventBus,
        settings: AudioSettings,
        *,
        session_start: float,
        backend: CaptureBackend | None = None,
        detector_factory: Callable[[], SpeechDetector] | None = None,
        watch_interval: float = 0.5,
    ) -> None:
        self._bus = bus
        self._settings = settings
        self._session_start = session_start
        self._backend: CaptureBackend = backend or WasapiBackend(
            loopback_device=settings.loopback_device,
            mic_device=settings.mic_device,
            frame_ms=settings.frame_ms,
        )
        self._detector_factory = detector_factory or (
            lambda: webrtc_detector(settings.vad_aggressiveness)
        )
        self._watch_interval = watch_interval
        self._running: list[tuple[_Channel, Capture]] = []
        self._watcher: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._running:
            raise RuntimeError("Audio source is already started")
        try:
            # Opening WASAPI devices blocks for tens of milliseconds: not on the event loop.
            await asyncio.to_thread(self._open_all)
        except AudioDeviceError as error:
            self._report(ComponentState.ERROR, str(error))
            raise
        self._report(ComponentState.OK)
        self._watcher = asyncio.create_task(self._watch(), name="audio-watch")

    async def stop(self) -> None:
        if self._watcher is not None:
            self._watcher.cancel()
            await asyncio.gather(self._watcher, return_exceptions=True)
            self._watcher = None
        await asyncio.to_thread(self._close_all)

    def _open_all(self) -> None:
        try:
            for speaker in _CHANNELS:
                device = self._backend.device(speaker)
                channel = _Channel(
                    speaker,
                    device,
                    self._settings,
                    # webrtcvad keeps state per stream: each channel needs its own.
                    self._detector_factory(),
                    session_start=self._session_start,
                    publish=self._bus.publish_threadsafe,
                )
                capture = self._backend.open(device, channel.on_audio)
                capture.start()
                self._running.append((channel, capture))
        except BaseException:
            # One channel is no use without the other: a half-open source would answer
            # without hearing the meeting, or without the user's side of it.
            self._close_all()
            raise

    def _close_all(self) -> None:
        running, self._running = self._running, []
        for _, capture in running:
            capture.stop()
        self._backend.close()

    async def _watch(self) -> None:
        while True:
            await asyncio.sleep(self._watch_interval)
            lost = [channel for channel, capture in self._running if not capture.is_active]
            if lost:
                names = ", ".join(f"{_KIND[c.speaker]} {c.device.name!r}" for c in lost)
                log.error("audio_device_lost", devices=names)
                self._report(ComponentState.ERROR, f"Audio device stopped: {names}")
                return

    def _report(self, state: ComponentState, detail: str = "") -> None:
        self._bus.publish(StatusChanged(component=Component.AUDIO, state=state, detail=detail))
