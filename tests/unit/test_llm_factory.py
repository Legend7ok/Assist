import pytest

from assist.llm.factory import create_llm_client
from assist.llm.openai_compat import OpenAICompatClient

LLM_ANSWER = """
[llm.answer]
provider = "openai_compat"
base_url = "https://api.groq.com/openai/v1"
model = "openai/gpt-oss-120b"
api_key_env = "GROQ_API_KEY"
"""


def test_builds_the_adapter_named_in_config(make_settings):
    settings = make_settings(toml=LLM_ANSWER, env="GROQ_API_KEY=secret-value\n")

    client = create_llm_client(settings, "answer")

    assert isinstance(client, OpenAICompatClient)


def test_missing_api_key_is_named_in_the_error(make_settings):
    settings = make_settings(toml=LLM_ANSWER)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        create_llm_client(settings, "answer")


def test_section_missing_from_config_is_named_in_the_error(make_settings):
    settings = make_settings(toml=LLM_ANSWER, env="GROQ_API_KEY=secret-value\n")

    with pytest.raises(ValueError, match=r"\[llm\.summary\]"):
        create_llm_client(settings, "summary")
