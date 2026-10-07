from collections.abc import AsyncIterator

import httpx2
import openai

from assist.core.schemas import Message
from assist.llm.base import LLMError
from assist.settings import LLMEndpoint


class OpenAICompatClient:
    """LLMClient for any provider with an OpenAI-compatible chat API (Groq, Gemini).

    Providers differ only in config: address, model, key and extra request parameters.
    """

    def __init__(
        self,
        endpoint: LLMEndpoint,
        api_key: str,
        *,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        self._model = endpoint.model
        self._params = endpoint.params
        self._client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint.base_url,
            timeout=endpoint.timeout,
            # The SDK retries 429 and dropped connections by itself, waiting seconds between
            # attempts. A live answer is useless that late, and the summarizer retries on its
            # own schedule, so a failure is reported at once.
            max_retries=0,
            http_client=http_client,
        )

    async def stream(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[message.model_dump() for message in messages],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                # Provider-specific options (reasoning_effort for gpt-oss on Groq) are sent
                # as they are, so a new option needs only a config change.
                extra_body=self._params,
            )
            # Closing releases the connection when the caller stops reading early, for
            # example when a new question cancels the answer being printed.
            async with response:
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except openai.APIError as error:
            raise LLMError(_describe(error)) from error


def _describe(error: openai.APIError) -> str:
    match error:
        case openai.RateLimitError():
            retry_after = error.response.headers.get("retry-after")
            hint = f", retry in {retry_after} s" if retry_after else ""
            return f"provider rate limit reached (HTTP 429){hint}"
        case openai.APIStatusError():
            detail = error.body.get("message") if isinstance(error.body, dict) else None
            suffix = f": {detail}" if detail else ""
            return f"provider returned HTTP {error.status_code}{suffix}"
        case openai.APITimeoutError():
            return "provider did not respond in time"
        case openai.APIConnectionError():
            return "connection to the provider failed or was cut off"
        case _:
            return f"provider error: {error.message}"
