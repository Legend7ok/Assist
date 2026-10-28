from typing import Literal

from assist.llm.base import LLMClient
from assist.llm.openai_compat import OpenAICompatClient
from assist.settings import Settings

type LLMPurpose = Literal["answer", "summary"]


def create_llm_client(settings: Settings, purpose: LLMPurpose) -> LLMClient:
    """Build the client for one purpose from its `[llm.<purpose>]` config section."""
    endpoint = getattr(settings.llm, purpose)
    if endpoint is None:
        raise ValueError(f"[llm.{purpose}] is not configured in config/default.toml")

    api_key = settings.api_key(endpoint.api_key_env)
    if api_key is None:
        raise ValueError(f"{endpoint.api_key_env} is not set in .env or the environment")

    match endpoint.provider:
        case "openai_compat":
            return OpenAICompatClient(endpoint, api_key.get_secret_value())
