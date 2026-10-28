import asyncio
from collections.abc import Callable

import structlog

from assist.core.bus import EventBus
from assist.core.events import (
    AudioFrame,
    Component,
    ComponentState,
    StatusChanged,
    TranscriptFinal,
    TranscriptPartial,
)
from assist.core.schemas import Segment
from assist.stt.base import Phrase, StereoFrame, STTError, Transcriber
from assist.stt.stereo import DEFAULT_BUFFER_MS, StereoAssembler

log = structlog.get_logger(__name__)


class TranscriptionService:
    """Speech from the bus to recognition, recognized phrases back to the bus.

    AudioFrame of both speakers are joined into stereo frames and sent to the transcriber.
    Its phrases come back as TranscriptPartial and TranscriptFinal with meeting times.
    StatusChanged for STT tells when recognition is connecting, working, or has failed and
    why; a failed transcriber is not restarted, the service has to be.
    """

    def __init__(
        self,
        bus: EventBus,
        transcriber: Transcriber,
        *,
        frame_ms: int,
        clock: Callable[[], float],
        buffer_ms: int = DEFAULT_BUFFER_MS,
    ) -> None:
        self._bus = bus
        self._transcriber = transcriber
        self._assembler = StereoAssembler(frame_ms=frame_ms, buffer_ms=buffer_ms)
        # Seconds since the session started, the clock of the phrases: for the latency log.
        self._clock = clock
        # Once no frame has come for as long as the buffer waits, the held frames have
        # nothing left to wait for.
        self._idle = buffer_ms / 1000
        self._last_frame = 0.0
        self._holding = asyncio.Event()
        # Frames go out in time order: the idle flush and new frames must not interleave.
        self._sending = asyncio.Lock()
        self._started = False
        self._stopped = False
        self._failed = False
        self._publisher: asyncio.Task[None] | None = None
        self._watcher: asyncio.Task[None] | None = None
        bus.subscribe(AudioFrame, self._on_frame, name="transcription")

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("Transcription is already started")
        self._started = True
        self._report(ComponentState.CONNECTING)
        try:
            await self._transcriber.connect()
        except STTError as error:
            self._fail(error)
            raise
        self._report(ComponentState.OK)
        self._publisher = asyncio.create_task(self._publish_phrases(), name="stt-phrases")
        self._watcher = asyncio.create_task(self._flush_when_idle(), name="stt-idle-flush")

    async def stop(self) -> None:
        """Send what is held, end the session and publish its last finals."""
        if not self._started or self._stopped:
            return
        self._stopped = True
        async with self._sending:
            # Under the lock the watcher is not in the middle of a send: cancelling it
            # cannot cut a frame short.
            if self._watcher is not None:
                self._watcher.cancel()
            await self._send(self._assembler.flush())
        if self._watcher is not None:
            await asyncio.gather(self._watcher, return_exceptions=True)
        await self._transcriber.close()
        if self._publisher is not None:
            # Ends by itself once the transcriber has delivered the last finals.
            await self._publisher

    async def _on_frame(self, frame: AudioFrame) -> None:
        if not self._started:
            return
        async with self._sending:
            # Checked under the lock: a frame that waited for it while stop() flushed must
            # not reach a closed transcriber, which would open a new session for it.
            if self._stopped or self._failed:
                return
            self._last_frame = asyncio.get_running_loop().time()
            self._holding.set()
            await self._send(self._assembler.add(frame))

    async def _flush_when_idle(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._holding.wait()
            wait = self._last_frame + self._idle - loop.time()
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            async with self._sending:
                # A frame may have come while this waited for the lock.
                if loop.time() - self._last_frame >= self._idle:
                    self._holding.clear()
                    await self._send(self._assembler.flush())

    async def _send(self, frames: list[StereoFrame]) -> None:
        for frame in frames:
            if self._failed:
                return
            try:
                await self._transcriber.send(frame)
            except STTError as error:
                self._fail(error)

    async def _publish_phrases(self) -> None:
        try:
            async for phrase in self._transcriber.phrases():
                self._publish(phrase)
        except STTError as error:
            self._fail(error)

    def _publish(self, phrase: Phrase) -> None:
        if not phrase.is_final:
            self._bus.publish(TranscriptPartial(speaker=phrase.speaker, text=phrase.text))
            return
        segment = Segment(
            speaker=phrase.speaker, text=phrase.text, start=phrase.start, end=phrase.end
        )
        log.info(
            "transcript_final",
            channel=phrase.speaker,
            # From the end of the speech to its text: what the user waits for.
            latency_ms=round((self._clock() - phrase.end) * 1000),
        )
        self._bus.publish(TranscriptFinal(segment=segment))

    def _fail(self, error: STTError) -> None:
        if self._failed:
            return
        self._failed = True
        self._report(ComponentState.ERROR, str(error))

    def _report(self, state: ComponentState, detail: str = "") -> None:
        self._bus.publish(StatusChanged(component=Component.STT, state=state, detail=detail))
