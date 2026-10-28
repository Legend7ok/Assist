"""Audio capture from WASAPI devices: the meeting from the loopback, the user from the mic."""

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import pyaudiowpatch as pyaudio
import structlog

from assist.core.schemas import Speaker

log = structlog.get_logger(__name__)

# PyAudioWPatch names the loopback twin of an output device "<output name> [Loopback]".
_LOOPBACK_SUFFIX = " [Loopback]"

type AudioCallback = Callable[[np.ndarray, float], None]


class AudioDeviceError(Exception):
    """A device cannot be found or opened. The message is written for the user."""


@dataclass(frozen=True, slots=True)
class AudioDevice:
    index: int
    name: str
    input_channels: int
    sample_rate: int
    is_loopback: bool

    @classmethod
    def from_info(cls, info: Mapping[str, Any]) -> Self:
        return cls(
            index=int(info["index"]),
            name=str(info["name"]),
            input_channels=int(info["maxInputChannels"]),
            sample_rate=int(info["defaultSampleRate"]),
            is_loopback=bool(info["isLoopbackDevice"]),
        )


@dataclass(frozen=True, slots=True)
class DeviceCatalog:
    """WASAPI devices of the system and the rules for picking one by the config.

    A configured name is matched case-insensitively: an exact name wins, otherwise the name
    must be a part of exactly one device name. No name means the Windows default device.
    """

    devices: tuple[AudioDevice, ...]
    default_input_index: int
    default_output_index: int

    @classmethod
    def from_pyaudio(cls, pa: pyaudio.PyAudio) -> Self:
        # Only WASAPI: loopback exists only there, and MME / DirectSound duplicates of the
        # same devices would make names ambiguous.
        try:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as error:
            raise AudioDeviceError("Windows WASAPI audio is not available") from error
        infos = pa.get_device_info_generator_by_host_api(host_api_index=wasapi["index"])
        return cls(
            devices=tuple(AudioDevice.from_info(info) for info in infos),
            default_input_index=wasapi["defaultInputDevice"],
            default_output_index=wasapi["defaultOutputDevice"],
        )

    def loopback(self, name: str | None) -> AudioDevice:
        candidates = [d for d in self.devices if d.is_loopback]
        if name is not None:
            return _match(candidates, name, kind="loopback")

        output = self._by_index(self.default_output_index)
        if output is None:
            raise AudioDeviceError("Windows has no default output device")
        # An exact name, not the substring check PyAudioWPatch uses: "Speakers" would also
        # match "Speakers 2 [Loopback]".
        twin = output.name + _LOOPBACK_SUFFIX
        for device in candidates:
            if device.name == twin:
                return device
        raise AudioDeviceError(f"No loopback device for the default output {output.name!r}")

    def microphone(self, name: str | None) -> AudioDevice:
        candidates = [d for d in self.devices if d.input_channels > 0 and not d.is_loopback]
        if name is not None:
            return _match(candidates, name, kind="microphone")

        device = self._by_index(self.default_input_index)
        if device is None or device not in candidates:
            raise AudioDeviceError("Windows has no default microphone")
        return device

    def _by_index(self, index: int) -> AudioDevice | None:
        # PortAudio reports a missing default device as -1.
        return next((d for d in self.devices if d.index == index), None)


def _match(candidates: list[AudioDevice], name: str, *, kind: str) -> AudioDevice:
    wanted = name.casefold()
    found = [d for d in candidates if d.name.casefold() == wanted] or [
        d for d in candidates if wanted in d.name.casefold()
    ]
    if len(found) == 1:
        return found[0]
    if not found:
        raise AudioDeviceError(
            f"No {kind} device matches {name!r}. Available: {_names(candidates)}"
        )
    raise AudioDeviceError(
        f"Several {kind} devices match {name!r}: {_names(found)}. Give a longer part of the name"
    )


def _names(devices: list[AudioDevice]) -> str:
    return ", ".join(repr(d.name) for d in devices) or "none"


class AudioCapture:
    """One input stream from a WASAPI device, delivered in blocks from a PortAudio thread.

    `on_audio` gets float32 samples shaped (frames, channels) in the device's own format, and
    the time.monotonic() moment the block started. It runs on the PortAudio thread, so it must
    be quick and must not touch the event loop directly.
    """

    def __init__(
        self,
        pa: pyaudio.PyAudio,
        device: AudioDevice,
        *,
        frame_ms: int,
        on_audio: AudioCallback,
    ) -> None:
        self._pa = pa
        self._device = device
        self._frames_per_buffer = device.sample_rate * frame_ms // 1000
        self._on_audio = on_audio
        self._stream: pyaudio.Stream | None = None
        self._failing = False

    @property
    def device(self) -> AudioDevice:
        return self._device

    @property
    def is_active(self) -> bool:
        """False once started and the device is gone (unplugged, disabled)."""
        return self._stream is not None and self._stream.is_active()

    def start(self) -> None:
        if self._stream is not None:
            raise RuntimeError("Audio capture is already started")
        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self._device.input_channels,
                rate=self._device.sample_rate,
                input=True,
                input_device_index=self._device.index,
                frames_per_buffer=self._frames_per_buffer,
                stream_callback=self._callback,
            )
        except OSError as error:
            raise AudioDeviceError(f"Cannot open {self._device.name!r}: {error}") from error
        log.info(
            "audio_capture_started",
            device=self._device.name,
            sample_rate=self._device.sample_rate,
            channels=self._device.input_channels,
        )

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            # Waits for a callback in progress, so on_audio is not called after stop().
            stream.stop_stream()
        except OSError:
            # A device that disappeared cannot be stopped cleanly; closing still frees it.
            log.warning("audio_capture_stop_failed", device=self._device.name, exc_info=True)
        stream.close()
        log.info("audio_capture_stopped", device=self._device.name)

    def _callback(
        self, in_data: bytes, frame_count: int, time_info: Mapping[str, float], status: int
    ) -> tuple[None, int]:
        # Not time_info: it is on PortAudio's own clock. time.monotonic() is one clock for
        # both channels and the rest of the app. The block arrives when its last sample is
        # recorded, so its start is one block length earlier.
        started_at = time.monotonic() - frame_count / self._device.sample_rate
        if status & pyaudio.paInputOverflow:
            log.warning("audio_input_overflow", device=self._device.name)
        try:
            samples = np.frombuffer(in_data, dtype=np.float32).reshape(
                -1, self._device.input_channels
            )
            self._on_audio(samples, started_at)
        except Exception:
            # An exception escaping to PortAudio aborts the stream for good, so a bad block is
            # dropped instead. A run of failures is logged once, not 50 times a second.
            if not self._failing:
                log.exception("audio_callback_failed", device=self._device.name)
            self._failing = True
        else:
            self._failing = False
        return None, pyaudio.paContinue


class WasapiBackend:
    """Opens the configured WASAPI device for each channel: the loopback for THEM, the mic for ME.

    PortAudio lists devices only once, when it starts. A device plugged in later shows up
    after close(), with the next start of the audio source.
    """

    def __init__(self, *, loopback_device: str | None, mic_device: str | None, frame_ms: int):
        self._names = {Speaker.THEM: loopback_device, Speaker.ME: mic_device}
        self._frame_ms = frame_ms
        self._pa: pyaudio.PyAudio | None = None

    def device(self, speaker: Speaker) -> AudioDevice:
        catalog = DeviceCatalog.from_pyaudio(self._portaudio())
        name = self._names[speaker]
        if speaker is Speaker.THEM:
            return catalog.loopback(name)
        return catalog.microphone(name)

    def open(self, device: AudioDevice, on_audio: AudioCallback) -> AudioCapture:
        return AudioCapture(self._portaudio(), device, frame_ms=self._frame_ms, on_audio=on_audio)

    def close(self) -> None:
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _portaudio(self) -> pyaudio.PyAudio:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa
