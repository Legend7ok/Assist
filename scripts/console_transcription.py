"""Console run of speech recognition: the transcript of a live call or of a recording.

Live: both channels from the devices in config/default.toml, as the app captures them.
Recording: them.wav and me.wav from record_sample.py, played in real time through the same
silence cutting as the app, so a run can be repeated without a call. Record them without
--speech-only: then the pauses keep their length and the times stay true.

The phrase being spoken is updated in place; a finished one is printed as
`[mm:ss] Speaker: text  (+latency)`, the latency being from the end of the speech to its text.
Needs GLADIA_API_KEY in .env. Ctrl+C stops; a recording also stops at its end.

Run: uv run python scripts/console_transcription.py
     uv run python scripts/console_transcription.py --wav [DIR]
"""

import argparse
import asyncio
import shutil
import sys
import time
import wave
from collections.abc import Awaitable
from pathlib import Path

from assist.audio.capture import AudioDeviceError
from assist.audio.resample import TARGET_RATE, PcmFrame
from assist.audio.source import AudioSource
from assist.audio.vad import SpeechFilter, webrtc_detector
from assist.core.bus import EventBus
from assist.core.events import (
    AudioFrame,
    ComponentState,
    Event,
    StatusChanged,
    TranscriptFinal,
    TranscriptPartial,
)
from assist.core.schemas import Speaker
from assist.logging_setup import configure_logging
from assist.settings import PROJECT_ROOT, AudioSettings, Settings
from assist.stt.base import STTError
from assist.stt.factory import create_transcriber
from assist.stt.service import TranscriptionService

DEFAULT_WAV_DIR = PROJECT_ROOT / "samples"
FILE_NAMES = {Speaker.THEM: "them.wav", Speaker.ME: "me.wav"}
LABELS = {Speaker.THEM: "Собеседник", Speaker.ME: "Я"}
# Moves the cursor to the line start and erases the line: the partial is redrawn in place.
CLEAR_LINE = "\r\x1b[K"


class MeetingClock:
    """Seconds since the meeting started, the clock of AudioFrame and the phrases."""

    def __init__(self) -> None:
        self.origin = time.monotonic()

    def restart(self) -> None:
        self.origin = time.monotonic()

    def __call__(self) -> float:
        return time.monotonic() - self.origin


class Console:
    def __init__(self, clock: MeetingClock, failed: asyncio.Event) -> None:
        self._clock = clock
        self._failed = failed

    async def show(self, event: Event) -> None:
        match event:
            case TranscriptPartial():
                line = f"... {LABELS[event.speaker]}: {event.text}"
                # A wrapped line cannot be redrawn in place: the end of the phrase is kept.
                width = shutil.get_terminal_size().columns - 1
                if len(line) > width:
                    line = "..." + line[-(width - 3) :]
                print(CLEAR_LINE + line, end="", flush=True)
            case TranscriptFinal():
                segment = event.segment
                seconds = int(segment.start)
                latency = self._clock() - segment.end
                print(
                    f"{CLEAR_LINE}[{seconds // 60:02d}:{seconds % 60:02d}] "
                    f"{LABELS[segment.speaker]}: {segment.text}  (+{latency:.1f} s)",
                    flush=True,
                )
            case StatusChanged():
                detail = f": {event.detail}" if event.detail else ""
                print(f"{CLEAR_LINE}[{event.component}: {event.state}{detail}]", flush=True)
                if event.state is ComponentState.ERROR:
                    self._failed.set()


def read_wav(path: Path) -> bytes:
    try:
        with wave.open(str(path), "rb") as wav:
            if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, TARGET_RATE):
                raise SystemExit(f"{path}: expected 16 kHz mono 16-bit, as record_sample.py writes")
            return wav.readframes(wav.getnframes())
    except (OSError, wave.Error) as error:
        raise SystemExit(f"Cannot read {path}: {error}") from error


async def play_recording(
    bus: EventBus, settings: AudioSettings, folder: Path, clock: MeetingClock
) -> None:
    """Publish the recording frame by frame in real time, cut to speech like the app does."""
    recordings = {speaker: read_wav(folder / name) for speaker, name in FILE_NAMES.items()}
    filters = {
        speaker: SpeechFilter(
            webrtc_detector(settings.vad_aggressiveness),
            frame_ms=settings.frame_ms,
            pre_speech_ms=settings.pre_speech_ms,
            min_speech_ms=settings.min_speech_ms,
            hangover_ms=settings.hangover_ms,
        )
        for speaker in FILE_NAMES
    }
    frame_seconds = settings.frame_ms / 1000
    size = TARGET_RATE * settings.frame_ms // 1000 * 2
    clock.restart()
    for n in range(max(len(pcm) for pcm in recordings.values()) // size):
        at = n * frame_seconds
        await asyncio.sleep(max(at - clock(), 0))
        for speaker, pcm in recordings.items():
            chunk = pcm[n * size : (n + 1) * size]
            if len(chunk) < size:
                continue
            for speech in filters[speaker].process(PcmFrame(pcm=chunk, start=at)):
                bus.publish(AudioFrame(speaker=speaker, pcm=speech.pcm, start=speech.start))


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    configure_logging(settings.log.level)
    transcriber = create_transcriber(settings)
    if args.wav is not None:
        # Fails here, before a Gladia session is spent on a missing file.
        for name in FILE_NAMES.values():
            read_wav(args.wav / name)

    clock = MeetingClock()
    failed = asyncio.Event()
    bus = EventBus()
    service = TranscriptionService(bus, transcriber, frame_ms=settings.audio.frame_ms, clock=clock)
    bus.subscribe(Event, Console(clock, failed).show, name="console")
    await bus.start()
    source: AudioSource | None = None
    try:
        await service.start()
        if args.wav is not None:
            print(f"Playing {args.wav}, Ctrl+C to stop...")
            await first_of(play_recording(bus, settings.audio, args.wav, clock), failed.wait())
        else:
            clock.restart()
            source = AudioSource(bus, settings.audio, session_start=clock.origin)
            await source.start()
            print("Listening, Ctrl+C to stop...")
            await failed.wait()
    except (STTError, AudioDeviceError):
        pass  # already printed as a status line
    finally:
        if source is not None:
            await source.stop()
        print(f"{CLEAR_LINE}Stopping, waiting for the last phrases...", flush=True)
        await service.stop()
        await bus.close()


async def first_of(*awaitables: Awaitable[object]) -> None:
    tasks = [asyncio.ensure_future(a) for a in awaitables]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--wav",
        type=Path,
        nargs="?",
        const=DEFAULT_WAV_DIR,
        help=f"play them.wav and me.wav from this folder instead (default {DEFAULT_WAV_DIR})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    except ValueError as error:
        # Missing key or config section: the message names it, a traceback adds nothing.
        sys.exit(str(error))


if __name__ == "__main__":
    main()
