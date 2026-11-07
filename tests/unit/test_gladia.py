import asyncio
import json
from collections.abc import AsyncIterator

import httpx2
import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from assist.core.schemas import Speaker
from assist.settings import STTSettings
from assist.stt.base import Phrase, StereoFrame, STTError
from assist.stt.gladia import GladiaTranscriber

SETTINGS = STTSettings(provider="gladia", api_key_env="GLADIA_API_KEY")
FRAME_BYTES = 320 * 4  # 20 ms, two channels of int16


class FakeSocket:
    """A Gladia live connection: the test plays the server side."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.sent: list[str | bytes] = []
        self._incoming: asyncio.Queue[dict | int] = asyncio.Queue()
        self._close_code: int | None = None

    def server_sends(self, message: dict) -> None:
        self._incoming.put_nowait(message)

    def server_closes(self, code: int) -> None:
        self._incoming.put_nowait(code)

    async def send(self, message: str | bytes) -> None:
        if self._close_code is not None:
            raise _closed(self._close_code)
        self.sent.append(message)
        if isinstance(message, str) and json.loads(message)["type"] == "stop_recording":
            self.on_stop()

    def on_stop(self) -> None:
        self.server_closes(1000)

    async def recv(self) -> str:
        item = await self._incoming.get()
        if isinstance(item, int):
            self._close_code = item
            raise _closed(item)
        return json.dumps(item)

    async def close(self) -> None:
        self._close_code = self._close_code or 1000

    def audio(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


def _closed(code: int) -> Exception:
    if code == 1000:
        return ConnectionClosedOK(Close(code, ""), None)
    return ConnectionClosedError(Close(code, ""), None)


class FakeGladia:
    """The session endpoint and the WebSocket server behind it."""

    def __init__(self, *, status: int = 201, failing_connects: int = 0) -> None:
        self.status = status
        self.failing_connects = failing_connects
        self.requests: list[httpx2.Request] = []
        self.sockets: list[FakeSocket] = []
        self.connects: list[str] = []
        # While cleared, connecting hangs, like a network that is still down.
        self.network = asyncio.Event()
        self.network.set()

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        n = len(self.requests)
        if self.status != 201:
            return httpx2.Response(self.status, json={"message": "refused"})
        return httpx2.Response(201, json={"id": f"s{n}", "url": f"wss://gladia.test/{n}"})

    async def open_socket(self, url: str) -> FakeSocket:
        await self.network.wait()
        self.connects.append(url)
        if self.failing_connects:
            self.failing_connects -= 1
            raise OSError("network is unreachable")
        self.sockets.append(FakeSocket(url))
        return self.sockets[-1]

    def transcriber(self) -> GladiaTranscriber:
        return GladiaTranscriber(
            SETTINGS,
            "test-key",
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(self.handle)),
            open_socket=self.open_socket,
            reconnect_attempts=3,
            reconnect_delay=0,
            stop_timeout=1,
        )


def transcript(
    phrase_id: str,
    text: str,
    *,
    start: float,
    end: float,
    channel: int = 0,
    final: bool = True,
    session: str = "s1",
) -> dict:
    return {
        "type": "transcript",
        "session_id": session,
        "created_at": "2026-10-28T10:00:00Z",
        "data": {
            "id": phrase_id,
            "is_final": final,
            "utterance": {
                "text": text,
                "start": start,
                "end": end,
                "language": "en",
                "channel": channel,
            },
        },
    }


async def speech(transcriber: GladiaTranscriber, start: float, seconds: float) -> None:
    for n in range(round(seconds / 0.02)):
        await transcriber.send(StereoFrame(pcm=bytes(FRAME_BYTES), start=start + n * 0.02))


async def next_phrases(phrases: AsyncIterator[Phrase], count: int) -> list[Phrase]:
    return [await asyncio.wait_for(anext(phrases), timeout=1) for _ in range(count)]


async def settle() -> None:
    for _ in range(20):
        await asyncio.sleep(0)


@pytest.fixture
async def gladia():
    return FakeGladia()


@pytest.fixture
async def connected(gladia):
    transcriber = gladia.transcriber()
    await transcriber.connect()
    yield transcriber
    await transcriber.close()


async def test_session_asks_for_two_channels_and_the_configured_languages(gladia, connected):
    (request,) = gladia.requests
    body = json.loads(request.content)

    assert str(request.url) == "https://api.gladia.io/v2/live"
    assert request.headers["x-gladia-key"] == "test-key"
    assert body["encoding"] == "wav/pcm"
    assert (body["sample_rate"], body["bit_depth"], body["channels"]) == (16000, 16, 2)
    assert body["model"] == "solaria-1"
    assert body["language_config"] == {"languages": ["en", "uk"], "code_switching": True}
    assert body["endpointing"] == SETTINGS.endpointing_seconds
    assert body["maximum_duration_without_endpointing"] == SETTINGS.max_phrase_seconds
    assert body["messages_config"]["receive_partial_transcripts"] is True
    assert gladia.connects == ["wss://gladia.test/1"]


async def test_audio_goes_as_binary_frames(gladia, connected):
    await speech(connected, 10.0, 0.1)

    assert gladia.sockets[0].audio() == [bytes(FRAME_BYTES)] * 5


async def test_partial_is_followed_by_the_final_of_the_same_phrase(gladia, connected):
    await speech(connected, 10.0, 2.0)
    socket = gladia.sockets[0]

    socket.server_sends(transcript("00_1", "What is", start=0.2, end=0.6, final=False))
    socket.server_sends(transcript("00_1", "What is an index?", start=0.2, end=1.4))
    partial, final = await next_phrases(connected.phrases(), 2)

    assert (partial.text, partial.is_final) == ("What is", False)
    assert (final.text, final.is_final) == ("What is an index?", True)
    assert partial.id == final.id


async def test_channel_gives_the_speaker(gladia, connected):
    await speech(connected, 10.0, 1.0)
    socket = gladia.sockets[0]

    socket.server_sends(transcript("00_1", "question", start=0.1, end=0.5, channel=0))
    socket.server_sends(transcript("00_2", "answer", start=0.5, end=0.9, channel=1))
    them, me = await next_phrases(connected.phrases(), 2)

    assert (them.speaker, me.speaker) == (Speaker.THEM, Speaker.ME)


async def test_times_are_moved_to_the_meeting_clock(gladia, connected):
    # 1 s of speech at 10 s, the pause after it cut, 1 s more at 40 s.
    await speech(connected, 10.0, 1.0)
    await speech(connected, 40.0, 1.0)

    gladia.sockets[0].server_sends(transcript("00_1", "later", start=1.2, end=1.5))
    (phrase,) = await next_phrases(connected.phrases(), 1)

    assert (phrase.start, phrase.end) == pytest.approx((40.2, 40.5))


async def test_empty_text_is_skipped_and_unknown_messages_ignored(gladia, connected):
    await speech(connected, 10.0, 1.0)
    socket = gladia.sockets[0]

    socket.server_sends({"type": "speech_start", "session_id": "s1", "data": {}})
    socket.server_sends(transcript("00_1", "  ", start=0.1, end=0.2))
    socket.server_sends(transcript("00_2", "text", start=0.3, end=0.5))
    (phrase,) = await next_phrases(connected.phrases(), 1)

    assert phrase.text == "text"


@pytest.mark.parametrize(
    ("status", "reason"),
    [(401, "API key"), (402, "credit"), (429, "one live session")],
)
async def test_refused_session_gives_a_clear_reason_without_retries(status, reason):
    gladia = FakeGladia(status=status)
    transcriber = gladia.transcriber()

    with pytest.raises(STTError, match=reason):
        await transcriber.connect()
    with pytest.raises(STTError, match=reason):
        await transcriber.send(StereoFrame(pcm=bytes(FRAME_BYTES), start=0.0))

    assert len(gladia.requests) == 1
    assert gladia.connects == []
    with pytest.raises(STTError, match=reason):
        await anext(transcriber.phrases())
    await transcriber.close()


async def test_dropped_connection_is_restored_to_the_same_session(gladia, connected):
    await speech(connected, 10.0, 1.0)
    gladia.sockets[0].server_closes(1006)
    await settle()

    await speech(connected, 11.0, 1.0)
    gladia.sockets[1].server_sends(transcript("00_1", "after", start=1.2, end=1.5))
    (phrase,) = await next_phrases(connected.phrases(), 1)

    assert gladia.connects == ["wss://gladia.test/1", "wss://gladia.test/1"]
    assert len(gladia.requests) == 1
    assert len(gladia.sockets[1].audio()) == 50
    # Still one session: its audio time goes on from where it was.
    assert (phrase.start, phrase.end) == pytest.approx((11.2, 11.5))


async def test_frames_wait_while_the_connection_is_restored(gladia, connected):
    await speech(connected, 10.0, 0.1)
    gladia.network.clear()
    gladia.failing_connects = 2
    gladia.sockets[0].server_closes(1011)
    await settle()

    sending = asyncio.create_task(speech(connected, 10.1, 0.1))
    await settle()
    assert not sending.done()
    gladia.network.set()
    await asyncio.wait_for(sending, timeout=1)

    assert len(gladia.connects) == 4
    assert len(gladia.sockets[0].audio()) == 5
    assert len(gladia.sockets[1].audio()) == 5


async def test_connection_that_cannot_be_restored_is_an_error(gladia, connected):
    await speech(connected, 10.0, 0.1)
    gladia.failing_connects = 3
    gladia.sockets[0].server_closes(1006)
    await settle()

    with pytest.raises(STTError, match="3 attempts"):
        await speech(connected, 10.1, 0.1)
    with pytest.raises(STTError, match="3 attempts"):
        await anext(connected.phrases())


async def test_session_ended_by_the_server_is_replaced_on_the_next_frame(gladia, connected):
    await speech(connected, 10.0, 1.0)
    gladia.sockets[0].server_closes(1000)
    await settle()

    await speech(connected, 100.0, 1.0)
    gladia.sockets[1].server_sends(transcript("00_1", "new", start=0.2, end=0.5, session="s2"))
    (phrase,) = await next_phrases(connected.phrases(), 1)

    assert len(gladia.requests) == 2
    assert gladia.connects[-1] == "wss://gladia.test/2"
    # The new session counts its audio from zero again.
    assert (phrase.start, phrase.end) == pytest.approx((100.2, 100.5))
    assert phrase.id == "s2:00_1"


async def test_close_stops_recording_and_delivers_the_last_finals(gladia):
    transcriber = gladia.transcriber()
    await transcriber.connect()
    await speech(transcriber, 10.0, 1.0)
    socket = gladia.sockets[0]

    def finish() -> None:
        socket.server_sends(transcript("00_1", "last words", start=0.1, end=0.9))
        socket.server_closes(1000)

    socket.on_stop = finish
    await transcriber.close()

    phrases = [phrase async for phrase in transcriber.phrases()]
    assert [p.text for p in phrases] == ["last words"]
    assert json.loads(socket.sent[-1]) == {"type": "stop_recording"}
    assert len(gladia.requests) == 1


async def test_close_does_not_wait_forever_for_a_silent_server(gladia):
    transcriber = gladia.transcriber()
    await transcriber.connect()
    gladia.sockets[0].on_stop = lambda: None

    await asyncio.wait_for(transcriber.close(), timeout=3)

    assert [phrase async for phrase in transcriber.phrases()] == []
