"""Console run of the assistant, before audio and the window exist.

Phrases are typed instead of recognized, a lone AltGr (or /ask) asks for an answer, and the
answer is printed piece by piece as it comes. Needs GROQ_API_KEY in .env; the summary works
once [llm.summary] is configured, until then each attempt to summarize fails visibly.

Run: uv run python scripts/console_assistant.py
To see the summary sooner, shrink the window, for example:
    ASSIST_CONTEXT__WINDOW_SECONDS=60 ASSIST_CONTEXT__COMPRESS_SECONDS=30
"""

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator

from assist.assistant.answerer import Answerer
from assist.context.notes import MeetingNotes
from assist.context.summarizer import Summarizer
from assist.context.window import ContextWindow
from assist.core.bus import EventBus
from assist.core.events import (
    AnswerDelta,
    AnswerDone,
    AnswerFailed,
    AnswerRequested,
    Event,
    NotesChanged,
    RequestChanged,
    SummaryFailed,
    SummaryRequested,
    SummaryUpdated,
    TranscriptFinal,
)
from assist.core.schemas import Message, Segment, Speaker
from assist.desktop.hotkeys import AnswerHotkey
from assist.llm.base import LLMClient, LLMError
from assist.llm.factory import create_llm_client
from assist.logging_setup import configure_logging
from assist.settings import Settings

HELP = """\
Commands:
  я: <text>          a phrase of mine
  он: <text>         a phrase of the interlocutor
  AltGr or /ask      answer the last question
  /notes <text>      meeting notes; /notes alone reads lines until /end
  /request <text>    one-off request; /request alone reads lines until /end (for code)
  /quit              exit
"""

SPEAKERS = {"я": Speaker.ME, "он": Speaker.THEM}


class UnavailableLLM:
    """Stands in for a model whose config section is missing: every call fails with why."""

    def __init__(self, reason: str) -> None:
        self._reason = reason

    async def stream(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        raise LLMError(self._reason)
        yield  # makes this an async generator, like the real clients


async def show(event: Event) -> None:
    match event:
        case AnswerRequested():
            print("\n=== answer ===", flush=True)
        case AnswerDelta():
            print(event.text, end="", flush=True)
        case AnswerDone():
            print("\n=== done ===", flush=True)
        case AnswerFailed():
            print(f"\n=== failed: {event.reason} ===", flush=True)
        case SummaryRequested():
            print(f"[summary: compressing {len(event.items)} items]", flush=True)
        case SummaryUpdated():
            print(f"[summary updated]\n{event.text}", flush=True)
        case SummaryFailed():
            print(f"[summary failed: {event.reason}]", flush=True)


def read_stdin_into(lines: asyncio.Queue[str | None], loop: asyncio.AbstractEventLoop) -> None:
    # A daemon thread and not asyncio.to_thread: input() cannot be interrupted, and the
    # default executor would keep the process waiting for Enter after /quit or Ctrl+C.
    def read() -> None:
        try:
            for line in sys.stdin:
                loop.call_soon_threadsafe(lines.put_nowait, line.rstrip("\n"))
            loop.call_soon_threadsafe(lines.put_nowait, None)
        except RuntimeError:
            pass  # the loop is closed: the script is exiting

    threading.Thread(target=read, name="stdin", daemon=True).start()


async def read_commands(bus: EventBus, clock) -> None:
    lines: asyncio.Queue[str | None] = asyncio.Queue()
    read_stdin_into(lines, asyncio.get_running_loop())
    # The command reading several lines, and the lines read so far.
    collecting: tuple[str, list[str]] | None = None

    while (line := await lines.get()) is not None:
        if collecting is not None:
            command, collected = collecting
            if line.strip() == "/end":
                publish_field(bus, command, "\n".join(collected))
                collecting = None
            else:
                collected.append(line)
            continue

        command, _, text = line.strip().partition(" ")
        match command:
            case "/quit":
                return
            case "/ask":
                bus.publish(AnswerRequested())
            case "/notes" | "/request" if text:
                publish_field(bus, command, text)
            case "/notes" | "/request":
                collecting = (command, [])
                print("... type the lines, finish with /end")
            case "":
                pass
            case _:
                publish_phrase(bus, clock, line)


def publish_field(bus: EventBus, command: str, text: str) -> None:
    if command == "/notes":
        bus.publish(NotesChanged(text=text))
        print("[notes set]")
    else:
        bus.publish(RequestChanged(text=text))
        print("[request set, press AltGr]")


def publish_phrase(bus: EventBus, clock, line: str) -> None:
    who, separator, text = line.partition(":")
    speaker = SPEAKERS.get(who.strip().lower())
    if not separator or speaker is None or not text.strip():
        print(HELP)
        return
    now = clock()
    segment = Segment(speaker=speaker, text=text.strip(), start=now, end=now)
    bus.publish(TranscriptFinal(segment=segment))


async def run() -> None:
    settings = Settings()
    configure_logging(settings.log.level)
    answer_client = create_llm_client(settings, "answer")
    summary_client: LLMClient
    try:
        summary_client = create_llm_client(settings, "summary")
    except ValueError as error:
        print(f"Summary is off: {error}")
        summary_client = UnavailableLLM(str(error))

    started = time.monotonic()

    def clock() -> float:
        return time.monotonic() - started

    bus = EventBus()
    window = ContextWindow(bus, settings.context)
    summarizer = Summarizer(bus, summary_client, settings.context)
    notes = MeetingNotes(bus)
    answerer = Answerer(
        bus,
        answer_client,
        settings.assistant,
        window=window,
        summarizer=summarizer,
        notes=notes,
        clock=clock,
    )
    bus.subscribe(Event, show, name="console")
    await bus.start()
    hotkey = AnswerHotkey(bus)
    hotkey.start()
    print(HELP)
    try:
        await read_commands(bus, clock)
    finally:
        hotkey.stop()
        await answerer.close()
        await bus.close()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    except ValueError as error:
        # Missing key or config section: the message names it, a traceback adds nothing.
        sys.exit(str(error))


if __name__ == "__main__":
    main()
