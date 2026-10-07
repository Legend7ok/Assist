import json
from collections.abc import AsyncIterator, Callable

import httpx2
import pytest

from assist.core.schemas import Message
from assist.llm.base import LLMError
from assist.llm.openai_compat import OpenAICompatClient
from assist.settings import LLMEndpoint

ENDPOINT = LLMEndpoint(
    provider="openai_compat",
    base_url="https://llm.test/v1",
    model="test-model",
    api_key_env="GROQ_API_KEY",
    params={"reasoning_effort": "low"},
)
MESSAGES = [Message(role="system", content="rules"), Message(role="user", content="question")]


def chunk(delta: dict, finish_reason: str | None = None) -> str:
    body = {
        "id": "chunk",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(body)}\n\n"


def sse_response(*events: str) -> httpx2.Response:
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content="".join(events).encode(),
    )


def make_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> OpenAICompatClient:
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return OpenAICompatClient(ENDPOINT, "test-key", http_client=http_client)


async def collect(client: OpenAICompatClient, received: list[str] | None = None) -> list[str]:
    received = [] if received is None else received
    async for piece in client.stream(MESSAGES, max_tokens=500, temperature=0.3):
        received.append(piece)
    return received


async def test_text_arrives_in_pieces_in_order():
    client = make_client(
        lambda _: sse_response(
            chunk({"role": "assistant"}),
            chunk({"content": "Use "}),
            chunk({"content": "an index"}),
            chunk({"content": "."}),
            chunk({}, finish_reason="stop"),
            "data: [DONE]\n\n",
        )
    )

    assert await collect(client) == ["Use ", "an index", "."]


async def test_request_carries_config_and_call_parameters():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return sse_response("data: [DONE]\n\n")

    await collect(make_client(handler))

    (request,) = requests
    body = json.loads(request.content)
    assert str(request.url) == "https://llm.test/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "test-model"
    assert body["messages"] == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "question"},
    ]
    assert body["max_tokens"] == 500
    assert body["temperature"] == 0.3
    assert body["stream"] is True
    assert body["reasoning_effort"] == "low"


async def test_rate_limit_gives_a_clear_error_at_once():
    calls = 0

    def handler(_: httpx2.Request) -> httpx2.Response:
        nonlocal calls
        calls += 1
        return httpx2.Response(
            429,
            headers={"retry-after": "7"},
            json={"error": {"message": "Rate limit reached", "type": "tokens"}},
        )

    with pytest.raises(LLMError, match=r"rate limit.*429.*retry in 7 s"):
        await collect(make_client(handler))
    assert calls == 1


async def test_rejected_request_names_status_and_provider_message():
    client = make_client(
        lambda _: httpx2.Response(401, json={"error": {"message": "Invalid API Key"}})
    )

    with pytest.raises(LLMError, match="HTTP 401: Invalid API Key"):
        await collect(client)


class _CutOffStream(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield chunk({"content": "First part"}).encode()
        raise httpx2.RemoteProtocolError("peer closed connection")


async def test_connection_cut_midway_keeps_received_text_and_gives_a_clear_error():
    client = make_client(
        lambda _: httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=_CutOffStream()
        )
    )
    received: list[str] = []

    with pytest.raises(LLMError, match="cut off"):
        await collect(client, received)
    assert received == ["First part"]
