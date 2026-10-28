import asyncio
import time
from collections.abc import Callable

import structlog

from assist.assistant.prompt import build_answer_prompt
from assist.context.notes import MeetingNotes
from assist.context.summarizer import Summarizer
from assist.context.window import ContextWindow
from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDelta,
    AnswerDone,
    AnswerFailed,
    AnswerRequested,
    RequestChanged,
)
from assist.core.schemas import Answer, new_id
from assist.llm.base import LLMClient, LLMError
from assist.settings import AssistantSettings

log = structlog.get_logger(__name__)


class Answerer:
    """Answers on AnswerRequested, streaming the text as AnswerDelta pieces.

    The prompt is built from the meeting context at the moment of the press. A new press while
    an answer is being written cancels it and starts over: the press means the user wants a
    different answer now, not the rest of the old one.
    """

    def __init__(
        self,
        bus: EventBus,
        client: LLMClient,
        settings: AssistantSettings,
        *,
        window: ContextWindow,
        summarizer: Summarizer,
        notes: MeetingNotes,
        clock: Callable[[], float],
    ) -> None:
        self._bus = bus
        self._client = client
        self._settings = settings
        self._window = window
        self._summarizer = summarizer
        self._notes = notes
        # Seconds since the session started, on the same clock as the phrases, so the answer
        # takes its place in the window's timeline.
        self._clock = clock
        self._running: tuple[asyncio.Task[None], str] | None = None
        # The answer runs in its own task: the handler has to stay free to see the next press.
        bus.subscribe(AnswerRequested, self._on_requested, name="answerer")

    async def close(self) -> None:
        await self._cancel_running()

    async def _on_requested(self, _: AnswerRequested) -> None:
        await self._cancel_running()
        answer_id = new_id()
        task = asyncio.create_task(self._answer(answer_id, self._clock()), name="answer")
        self._running = (task, answer_id)

    async def _cancel_running(self) -> None:
        if self._running is None:
            return
        task, answer_id = self._running
        self._running = None
        if task.done():
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        log.info("answer_cancelled", answer_id=answer_id)
        self._bus.publish(AnswerFailed(answer_id=answer_id, reason="cancelled by a new request"))

    async def _answer(self, answer_id: str, asked_at: float) -> None:
        request = self._notes.request
        messages = build_answer_prompt(
            notes=self._notes.notes,
            summary=self._summarizer.summary,
            items=self._window.items,
            request=request,
        )
        max_tokens = (
            self._settings.request_max_tokens if request else self._settings.answer_max_tokens
        )

        started = time.perf_counter()
        pieces: list[str] = []
        try:
            async for piece in self._client.stream(
                messages, max_tokens=max_tokens, temperature=self._settings.temperature
            ):
                if not pieces:
                    log.info("answer_first_piece", latency_ms=_ms_since(started))
                pieces.append(piece)
                self._bus.publish(AnswerDelta(answer_id=answer_id, text=piece))
            text = "".join(pieces).strip()
            if not text:
                # A reasoning model can spend the whole token cap on reasoning and say nothing.
                raise LLMError("model returned an empty answer")
        except LLMError as error:
            self._fail(answer_id, str(error))
            return
        except Exception:
            log.exception("answer_crashed", answer_id=answer_id)
            self._fail(answer_id, "internal error")
            return

        log.info("answer_done", chars=len(text), latency_ms=_ms_since(started))
        self._bus.publish(AnswerDone(answer=Answer(id=answer_id, text=text, at=asked_at)))
        # Cleared only if unchanged: a request typed while this answer was being written is
        # meant for the next one.
        if request and self._notes.request == request:
            self._bus.publish(RequestChanged(text=""))

    def _fail(self, answer_id: str, reason: str) -> None:
        log.warning("answer_failed", answer_id=answer_id, reason=reason)
        self._bus.publish(AnswerFailed(answer_id=answer_id, reason=reason))


def _ms_since(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)
