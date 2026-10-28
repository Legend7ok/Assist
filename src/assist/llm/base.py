from collections.abc import AsyncIterator
from typing import Protocol

from assist.core.schemas import Message


class LLMError(Exception):
    """A request to the model failed; the message says why in words a person can act on."""


class LLMClient(Protocol):
    def stream(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> AsyncIterator[str]:
        """Yield the answer text piece by piece as the model generates it.

        Raises LLMError when the provider refuses the request or the connection breaks,
        including midway through the answer, after some pieces were already yielded.
        """
        ...
