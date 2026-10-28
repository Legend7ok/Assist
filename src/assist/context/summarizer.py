import time

import structlog

from assist.assistant.prompts import load_prompt
from assist.context.window import format_timeline
from assist.core.bus import EventBus
from assist.core.events import (
    Component,
    ComponentState,
    StatusChanged,
    SummaryEdited,
    SummaryFailed,
    SummaryRequested,
    SummaryUpdated,
)
from assist.core.schemas import Message
from assist.llm.base import LLMClient, LLMError
from assist.settings import ContextSettings

log = structlog.get_logger(__name__)

# A Ukrainian word takes about 2-3 tokens; the cap only stops a runaway reply, the word limit
# in the prompt is what keeps the summary short.
_TOKENS_PER_WORD = 4
_TEMPERATURE = 0.2


class Summarizer:
    """Folds parts of the window into one running summary of the meeting.

    It has its own LLM client (Gemini), so the answer model's rate limit is left for answers.
    The summary is built on the previous one, including the user's edits to it.
    """

    def __init__(self, bus: EventBus, client: LLMClient, settings: ContextSettings) -> None:
        self._bus = bus
        self._client = client
        self._settings = settings
        self._instructions = load_prompt("summary").format(max_words=settings.summary_max_words)
        self._summary = ""
        self._busy = False
        bus.subscribe(SummaryRequested, self._on_requested, name="summarizer")
        # A separate subscription: an edit has to be seen while a summary is being written
        # instead of waiting behind it in the same queue.
        bus.subscribe(SummaryEdited, self._on_edited, name="summarizer_edits")

    @property
    def summary(self) -> str:
        return self._summary

    async def _on_requested(self, event: SummaryRequested) -> None:
        self._busy = True
        self._publish_status(ComponentState.BUSY)
        started = time.perf_counter()
        try:
            text = await self._summarize(event)
        except LLMError as error:
            self._fail(str(error))
            return
        except Exception:
            # The window waits for SummaryUpdated or SummaryFailed; without one of them it
            # would never hand over a part again.
            log.exception("summary_crashed")
            self._fail("internal error")
            return
        finally:
            self._busy = False

        self._summary = text
        log.info(
            "summary_updated",
            items=len(event.items),
            words=len(text.split()),
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        self._bus.publish(
            SummaryUpdated(text=text, absorbed_ids=tuple(item.id for item in event.items))
        )
        self._publish_status(ComponentState.OK)

    async def _summarize(self, event: SummaryRequested) -> str:
        messages = [
            Message(role="system", content=self._instructions),
            Message(
                role="user",
                content=(
                    f"Current summary:\n{self._summary or '(empty)'}\n\n"
                    f"Next part of the meeting:\n{format_timeline(event.items)}"
                ),
            ),
        ]
        pieces = [
            piece
            async for piece in self._client.stream(
                messages,
                max_tokens=self._settings.summary_max_words * _TOKENS_PER_WORD,
                temperature=_TEMPERATURE,
            )
        ]
        text = "".join(pieces).strip()
        if not text:
            # Absorbing the part into an empty summary would erase it from the context.
            raise LLMError("model returned an empty summary")
        return text

    async def _on_edited(self, event: SummaryEdited) -> None:
        if self._busy:
            # The summary being written is built on the old text and would overwrite the edit.
            log.warning("summary_edit_rejected", reason="summarizing")
            return
        self._summary = event.text

    def _fail(self, reason: str) -> None:
        log.warning("summary_failed", reason=reason)
        self._bus.publish(SummaryFailed(reason=reason))
        self._publish_status(ComponentState.ERROR, reason)

    def _publish_status(self, state: ComponentState, detail: str = "") -> None:
        self._bus.publish(
            StatusChanged(component=Component.LLM_SUMMARY, state=state, detail=detail)
        )
