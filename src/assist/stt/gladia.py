import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx2
import structlog
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from assist.settings import STTSettings
from assist.stt.base import (
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    STEREO_CHANNELS,
    Phrase,
    StereoFrame,
    STTError,
)
from assist.stt.timeline import Timeline

log = structlog.get_logger(__name__)

_SESSION_URL = "https://api.gladia.io/v2/live"
_HTTP_TIMEOUT = 10.0
_NORMAL_CLOSURE = 1000


class Socket(Protocol):
    async def send(self, message: str | bytes) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


type SocketOpener = Callable[[str], Awaitable[Socket]]


async def _open_socket(url: str) -> Socket:
    return await connect(url, open_timeout=_HTTP_TIMEOUT)


@dataclass(eq=False)
class _Session:
    url: str
    timeline: Timeline
    # None while the connection is being restored.
    socket: Socket | None


class GladiaTranscriber:
    """Transcriber on a Gladia live session: both speakers as the two channels of one stream.

    A dropped connection is restored to the same session, after a pause and a limited number
    of attempts; frames wait meanwhile. A session the server ends normally (code 1000) is
    replaced by a new one with the next frame. A refused session (wrong key, limit, credit)
    is reported at once: retrying cannot fix it.
    """

    def __init__(
        self,
        settings: STTSettings,
        api_key: str,
        *,
        http_client: httpx2.AsyncClient | None = None,
        open_socket: SocketOpener = _open_socket,
        reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
        stop_timeout: float = 10.0,
    ) -> None:
        self._settings = settings
        self._api_key = api_key
        self._owns_http = http_client is None
        self._http = http_client or httpx2.AsyncClient(timeout=_HTTP_TIMEOUT)
        self._open_socket = open_socket
        self._reconnect_attempts = reconnect_attempts
        self._reconnect_delay = reconnect_delay
        self._stop_timeout = stop_timeout
        self._session: _Session | None = None
        self._reader: asyncio.Task[None] | None = None
        self._error: STTError | None = None
        self._stopping = False
        # Senders wait on it while the reader restores or replaces the connection.
        self._changed = asyncio.Condition()
        self._starting = asyncio.Lock()
        self._phrases: asyncio.Queue[Phrase | STTError | None] = asyncio.Queue()

    async def connect(self) -> None:
        async with self._starting:
            if self._session is not None or self._error is not None:
                return
            try:
                await self._start_session()
            except STTError as error:
                # Kept, so frames already waiting do not ask Gladia for a session one by one:
                # a refusal stays a refusal.
                await self._fail(error)
                raise

    async def send(self, frame: StereoFrame) -> None:
        while True:
            if self._error is not None:
                raise self._error
            session = self._session
            if session is None:
                await self.connect()
                continue
            socket = session.socket
            if socket is not None:
                try:
                    await socket.send(frame.pcm)
                except ConnectionClosed:
                    pass
                else:
                    session.timeline.add(frame)
                    return
            # The connection is being restored, or has just closed and the reader is about to
            # restore it or end the session.
            await self._wait_for_change(session, socket)

    async def phrases(self) -> AsyncIterator[Phrase]:
        while (item := await self._phrases.get()) is not None:
            if isinstance(item, STTError):
                raise item
            yield item

    async def close(self) -> None:
        self._stopping = True
        session = self._session
        if session is not None and session.socket is not None:
            with contextlib.suppress(ConnectionClosed):
                await session.socket.send(json.dumps({"type": "stop_recording"}))
        if self._reader is not None:
            # The server sends the last finals and then closes the connection itself.
            done, _ = await asyncio.wait({self._reader}, timeout=self._stop_timeout)
            if not done:
                log.warning("stt_stop_timed_out", timeout=self._stop_timeout)
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
        if session is not None and session.socket is not None:
            await session.socket.close()
        if self._owns_http:
            await self._http.aclose()
        self._phrases.put_nowait(None)

    async def _start_session(self) -> None:
        url = await self._create_session()
        socket = await self._connect(url, reconnect=False)
        session = _Session(url, Timeline(), socket=socket)
        async with self._changed:
            self._session = session
            self._changed.notify_all()
        self._reader = asyncio.create_task(self._read(session, socket), name="gladia-reader")

    async def _create_session(self) -> str:
        try:
            response = await self._http.post(
                _SESSION_URL, headers={"x-gladia-key": self._api_key}, json=self._config()
            )
        except httpx2.HTTPError as error:
            raise STTError(f"Cannot reach Gladia: {error}") from error
        if not response.is_success:
            raise STTError(_describe_refusal(response))
        try:
            body = response.json()
            url = body["url"]
        except (ValueError, KeyError, TypeError) as error:
            raise STTError("Gladia created a session but gave no address for it") from error
        log.info("stt_session_created", session_id=body.get("id"))
        return url

    def _config(self) -> dict[str, Any]:
        settings = self._settings
        return {
            "encoding": "wav/pcm",
            "bit_depth": SAMPLE_WIDTH * 8,
            "sample_rate": SAMPLE_RATE,
            "channels": len(STEREO_CHANNELS),
            "model": "solaria-1",
            "endpointing": settings.endpointing_seconds,
            "maximum_duration_without_endpointing": settings.max_phrase_seconds,
            "language_config": {
                "languages": settings.languages,
                "code_switching": settings.code_switching,
            },
            # Only what is used: by default Gladia also acknowledges every audio chunk,
            # 50 messages a second for nothing.
            "messages_config": {
                "receive_partial_transcripts": True,
                "receive_final_transcripts": True,
                "receive_speech_events": False,
                "receive_pre_processing_events": False,
                "receive_realtime_processing_events": False,
                "receive_post_processing_events": False,
                "receive_acknowledgments": False,
                "receive_errors": True,
                "receive_lifecycle_events": False,
            },
        }

    async def _connect(self, url: str, *, reconnect: bool) -> Socket:
        for attempt in range(1, self._reconnect_attempts + 1):
            if reconnect:
                await asyncio.sleep(self._reconnect_delay)
            if self._stopping:
                raise STTError("Recognition stopped while reconnecting")
            try:
                return await self._open_socket(url)
            except (OSError, TimeoutError, InvalidHandshake) as error:
                log.warning("stt_connect_failed", attempt=attempt, error=str(error))
            reconnect = True
        raise STTError(
            f"Lost connection to Gladia: {self._reconnect_attempts} attempts to connect failed"
        )

    async def _read(self, session: _Session, socket: Socket) -> None:
        try:
            await self._serve(session, socket)
        except Exception:
            # Senders wait on the reader to restore the connection; a reader dead without a
            # word would leave them waiting forever.
            log.exception("stt_reader_crashed")
            await self._fail(STTError("Recognition failed: internal error"))

    async def _serve(self, session: _Session, socket: Socket) -> None:
        while True:
            code = await self._receive_until_closed(session, socket)
            if code == _NORMAL_CLOSURE or self._stopping:
                # Gladia closes a session normally after stop_recording, and it may end one
                # on its own (the 3-hour limit, maybe a long pause): the next frame opens a
                # new session.
                log.info(
                    "stt_session_ended",
                    by_server=not self._stopping,
                    audio_seconds=round(session.timeline.sent_seconds, 1),
                )
                await self._update(session, ended=True)
                return

            log.warning("stt_connection_lost", code=code)
            await self._update(session, socket=None)
            try:
                socket = await self._connect(session.url, reconnect=True)
            except STTError as error:
                if self._stopping:
                    await self._update(session, ended=True)
                else:
                    await self._fail(error)
                return
            log.info("stt_reconnected")
            await self._update(session, socket=socket)

    async def _receive_until_closed(self, session: _Session, socket: Socket) -> int | None:
        """Handle messages until the connection closes; return the close code, if any came."""
        try:
            while True:
                self._handle(await socket.recv(), session)
        except ConnectionClosed as closed:
            return closed.rcvd.code if closed.rcvd is not None else None

    def _handle(self, raw: str | bytes, session: _Session) -> None:
        try:
            message = json.loads(raw)
            if message.get("type") != "transcript":
                return
            phrase = self._phrase(message, session.timeline)
        except (ValueError, KeyError, TypeError, IndexError):
            # One malformed message must not end the session.
            log.exception("stt_message_unreadable")
            return
        if phrase is not None:
            self._phrases.put_nowait(phrase)

    def _phrase(self, message: dict[str, Any], timeline: Timeline) -> Phrase | None:
        data = message["data"]
        utterance = data["utterance"]
        text = utterance["text"].strip()
        if not text:
            return None
        start, end = timeline.span(utterance["start"], utterance["end"])
        return Phrase(
            # Gladia numbers phrases within a session; a new session starts the count again.
            id=f"{message['session_id']}:{data['id']}",
            speaker=STEREO_CHANNELS[utterance.get("channel", 0)],
            text=text,
            start=start,
            end=end,
            is_final=data["is_final"],
        )

    async def _update(
        self, session: _Session, *, socket: Socket | None = None, ended: bool = False
    ) -> None:
        async with self._changed:
            session.socket = socket
            if ended and self._session is session:
                self._session = None
            self._changed.notify_all()

    async def _fail(self, error: STTError) -> None:
        log.error("stt_failed", reason=str(error))
        async with self._changed:
            self._error = error
            if self._session is not None:
                self._session.socket = None
            self._session = None
            self._changed.notify_all()
        self._phrases.put_nowait(error)

    async def _wait_for_change(self, session: _Session, socket: Socket | None) -> None:
        async with self._changed:
            await self._changed.wait_for(
                lambda: (
                    session.socket is not socket
                    or self._session is not session
                    or self._error is not None
                )
            )


def _describe_refusal(response: httpx2.Response) -> str:
    match response.status_code:
        case 401:
            return "Gladia rejected the API key (HTTP 401)"
        case 402:
            return "Gladia credit is used up (HTTP 402)"
        case 429:
            return (
                "Gladia limit reached: the free plan allows one live session at a time (HTTP 429)"
            )
    try:
        body = response.json()
    except ValueError:
        body = None
    detail = body.get("message") if isinstance(body, dict) else None
    suffix = f": {detail}" if detail else ""
    return f"Gladia refused the session with HTTP {response.status_code}{suffix}"
