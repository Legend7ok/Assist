import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from assist.core.bus import EventBus
from assist.core.events import (
    AudioFrame,
    Component,
    ComponentState,
    Event,
    StatusChanged,
    TranscriptFinal,
    TranscriptPartial,
)
from assist.core.schemas import Speaker
from assist.stt.base import Phrase, StereoFrame, STTError
from assist.stt.service import TranscriptionService

SAMPLES = 320  # 20 ms at 16 kHz
BUFFER_MS = 40


class FakeTranscriber:
    """Records what it is sent; the test plays the phrases recognition gives back."""

    def __init__(self, *, connect_error: STTError | None = None) -> None:
        self.connect_error = connect_error
        self.send_error: STTError | None = None
        self.sent: list[StereoFrame] = []
        self.closed = False
        # Played on close(), like the finals a real session delivers after stop_recording.
        self.last_finals: list[Phrase] = []
        self._phrases: asyncio.Queue[Phrase | STTError | None] = asyncio.Queue()

    def recognizes(self, item: Phrase | STTError) -> None:
        self._phrases.put_nowait(item)

    async def connect(self) -> None:
        if self.connect_error is not None:
            raise self.connect_error

    async def send(self, frame: StereoFrame) -> None:
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(frame)

    async def phrases(self) -> AsyncIterator[Phrase]:
        while (item := await self._phrases.get()) is not None:
            if isinstance(item, STTError):
                raise item
            yield item

    async def close(self) -> None:
        self.closed = True
        for phrase in self.last_finals:
            self._phrases.put_nowait(phrase)
        self._phrases.put_nowait(None)


class Recognition:
    """The service on a running bus, with every event published on it."""

    def __init__(self, bus: EventBus, transcriber: FakeTranscriber) -> None:
        self.bus = bus
        self.transcriber = transcriber
        self.service = TranscriptionService(
            bus, transcriber, frame_ms=20, clock=lambda: 100.0, buffer_ms=BUFFER_MS
        )
        self.events: list[Event] = []

        async def record(event: Event) -> None:
            self.events.append(event)

        bus.subscribe(Event, record)

    async def speak(self, speaker: Speaker, start: float, count: int, value: int) -> None:
        await self.speak_together({speaker: value}, start, count)

    async def speak_together(self, values: dict[Speaker, int], start: float, count: int) -> None:
        """Publish frames of the speakers in the order two devices deliver them, each frame
        filled with the speaker's value."""
        for n in range(count):
            for speaker, value in values.items():
                pcm = np.full(SAMPLES, value, dtype=np.int16).tobytes()
                self.bus.publish(AudioFrame(speaker=speaker, pcm=pcm, start=start + n * 0.02))
        await settle()

    def of_type[E: Event](self, event_type: type[E]) -> list[E]:
        return [event for event in self.events if isinstance(event, event_type)]

    def states(self) -> list[tuple[ComponentState, str]]:
        return [(s.state, s.detail) for s in self.of_type(StatusChanged)]

    def sent_channels(self) -> list[tuple[int, int]]:
        return [
            tuple(int(v) for v in np.frombuffer(f.pcm, dtype=np.int16).reshape(-1, 2)[0])
            for f in self.transcriber.sent
        ]


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
async def make_recognition():
    started: list[Recognition] = []

    async def factory(transcriber: FakeTranscriber | None = None) -> Recognition:
        recognition = Recognition(EventBus(), transcriber or FakeTranscriber())
        await recognition.bus.start()
        started.append(recognition)
        return recognition

    yield factory
    for recognition in started:
        await recognition.service.stop()
        await recognition.bus.close()


def phrase(text: str, *, final: bool = True, speaker: Speaker = Speaker.THEM) -> Phrase:
    return Phrase(id="s1:00_1", speaker=speaker, text=text, start=12.0, end=13.5, is_final=final)


async def test_start_reports_connecting_then_working(make_recognition):
    recognition = await make_recognition()

    await recognition.service.start()
    await settle()

    assert recognition.states() == [(ComponentState.CONNECTING, ""), (ComponentState.OK, "")]
    assert all(s.component is Component.STT for s in recognition.of_type(StatusChanged))


async def test_refused_connection_is_reported_with_its_reason(make_recognition):
    refused = FakeTranscriber(connect_error=STTError("Gladia rejected the API key (HTTP 401)"))
    recognition = await make_recognition(refused)

    with pytest.raises(STTError):
        await recognition.service.start()
    await settle()

    assert recognition.states()[-1] == (
        ComponentState.ERROR,
        "Gladia rejected the API key (HTTP 401)",
    )


async def test_both_speakers_go_out_together_as_stereo(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()

    await recognition.speak_together({Speaker.THEM: 1, Speaker.ME: 2}, 10.0, 5)
    await asyncio.sleep(BUFFER_MS / 1000 * 3)

    assert recognition.sent_channels() == [(1, 2)] * 5
    assert [f.start for f in recognition.transcriber.sent] == pytest.approx(
        [10.0, 10.02, 10.04, 10.06, 10.08]
    )


async def test_held_frames_go_out_once_speech_stops(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()

    await recognition.speak(Speaker.THEM, 10.0, 10, value=1)
    # The buffer still holds the last two frames, waiting for the other speaker.
    assert len(recognition.transcriber.sent) == 8

    await asyncio.sleep(BUFFER_MS / 1000 * 3)

    assert len(recognition.transcriber.sent) == 10


async def test_partial_and_final_phrases_reach_the_bus(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()

    recognition.transcriber.recognizes(phrase("What is", final=False))
    recognition.transcriber.recognizes(phrase("What is an index?"))
    await settle()

    assert recognition.of_type(TranscriptPartial) == [
        TranscriptPartial(speaker=Speaker.THEM, text="What is")
    ]
    (final,) = recognition.of_type(TranscriptFinal)
    assert final.segment.speaker is Speaker.THEM
    assert final.segment.text == "What is an index?"
    assert (final.segment.start, final.segment.end) == (12.0, 13.5)


async def test_failed_sending_is_reported_once_and_sending_stops(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()
    recognition.transcriber.send_error = STTError("Lost connection to Gladia")

    await recognition.speak(Speaker.THEM, 10.0, 20, value=1)
    recognition.transcriber.send_error = None
    await recognition.speak(Speaker.THEM, 20.0, 20, value=1)

    errors = [s for s in recognition.states() if s[0] is ComponentState.ERROR]
    assert errors == [(ComponentState.ERROR, "Lost connection to Gladia")]
    assert recognition.transcriber.sent == []


async def test_failure_while_recognizing_is_reported(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()

    recognition.transcriber.recognizes(STTError("Lost connection to Gladia"))
    await settle()

    assert recognition.states()[-1] == (ComponentState.ERROR, "Lost connection to Gladia")


async def test_stop_sends_what_is_held_and_publishes_the_last_finals(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()
    await recognition.speak(Speaker.ME, 10.0, 10, value=2)
    recognition.transcriber.last_finals = [phrase("last words", speaker=Speaker.ME)]

    await recognition.service.stop()
    await settle()

    assert len(recognition.transcriber.sent) == 10
    assert recognition.transcriber.closed
    (final,) = recognition.of_type(TranscriptFinal)
    assert final.segment.text == "last words"


async def test_frames_after_stop_are_not_sent(make_recognition):
    recognition = await make_recognition()
    await recognition.service.start()
    await recognition.service.stop()

    await recognition.speak(Speaker.THEM, 10.0, 10, value=1)
    await asyncio.sleep(BUFFER_MS / 1000 * 3)

    assert recognition.transcriber.sent == []
