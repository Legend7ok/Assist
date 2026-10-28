"""Record both audio channels to WAV files, in the format speech recognition gets.

Usage: uv run python scripts/record_sample.py [--seconds 60] [--speech-only] [--out DIR]

Writes them.wav (the loopback) and me.wav (the microphone), 16 kHz mono int16, through the
same capture and resampling the app uses. With --speech-only the silence between phrases is
cut the way the app cuts it before recognition. Ctrl+C stops early and keeps what is recorded.
"""

import argparse
import time
import wave
from pathlib import Path

import numpy as np

from assist.audio.capture import AudioDeviceError, WasapiBackend
from assist.audio.resample import TARGET_RATE, FrameResampler
from assist.audio.vad import SpeechFilter, webrtc_detector
from assist.core.schemas import Speaker
from assist.logging_setup import configure_logging
from assist.settings import PROJECT_ROOT, Settings

# Inside the project, where .gitignore keeps the private recordings out of git.
DEFAULT_OUT = PROJECT_ROOT / "samples"
FILE_NAMES = {Speaker.THEM: "them.wav", Speaker.ME: "me.wav"}


class ChannelRecorder:
    """Collects one channel in memory: it is written to disk only after the capture stops,
    so the PortAudio thread never waits on the disk."""

    def __init__(self, sample_rate: int, speech_filter: SpeechFilter | None, frame_ms: int):
        self._resampler = FrameResampler(sample_rate, frame_ms=frame_ms)
        self._filter = speech_filter
        self._pcm = bytearray()

    def on_audio(self, samples: np.ndarray, started_at: float) -> None:
        for frame in self._resampler.process(samples, started_at):
            kept = [frame] if self._filter is None else self._filter.process(frame)
            for f in kept:
                self._pcm += f.pcm

    def save(self, path: Path) -> float:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(TARGET_RATE)
            wav.writeframes(self._pcm)
        return len(self._pcm) / 2 / TARGET_RATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=60.0, help="how long to record")
    parser.add_argument("--speech-only", action="store_true", help="cut silence like the app does")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    configure_logging(settings.log.level)
    audio = settings.audio
    backend = WasapiBackend(
        loopback_device=audio.loopback_device,
        mic_device=audio.mic_device,
        frame_ms=audio.frame_ms,
    )

    recorders: dict[Speaker, ChannelRecorder] = {}
    captures = []
    try:
        for speaker in (Speaker.THEM, Speaker.ME):
            device = backend.device(speaker)
            speech_filter = None
            if args.speech_only:
                speech_filter = SpeechFilter(
                    webrtc_detector(audio.vad_aggressiveness),
                    frame_ms=audio.frame_ms,
                    pre_speech_ms=audio.pre_speech_ms,
                    min_speech_ms=audio.min_speech_ms,
                    hangover_ms=audio.hangover_ms,
                )
            recorder = ChannelRecorder(device.sample_rate, speech_filter, audio.frame_ms)
            recorders[speaker] = recorder
            capture = backend.open(device, recorder.on_audio)
            capture.start()
            captures.append(capture)

        print(f"Recording {args.seconds:g} s, Ctrl+C to stop early...")
        deadline = time.monotonic() + args.seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopped early.")
    except AudioDeviceError as error:
        raise SystemExit(f"Audio device error: {error}") from error
    finally:
        for capture in captures:
            capture.stop()
        backend.close()

    args.out.mkdir(parents=True, exist_ok=True)
    for speaker, recorder in recorders.items():
        path = args.out / FILE_NAMES[speaker]
        seconds = recorder.save(path)
        print(f"{path}: {seconds:.1f} s")


if __name__ == "__main__":
    main()
