import pytest
from pydantic import ValidationError

LLM_ANSWER = """
[llm.answer]
provider = "openai_compat"
base_url = "https://api.groq.com/openai/v1"
model = "openai/gpt-oss-120b"
api_key_env = "{key}"
params = {{ reasoning_effort = "low" }}
"""


def test_reads_values_from_config_file(make_settings):
    settings = make_settings(toml='[log]\nlevel = "DEBUG"\n')

    assert settings.log.level == "DEBUG"


def test_environment_variable_overrides_config_file(make_settings, monkeypatch):
    monkeypatch.setenv("ASSIST_LOG__LEVEL", "ERROR")

    settings = make_settings(toml='[log]\nlevel = "DEBUG"\n')

    assert settings.log.level == "ERROR"


def test_starts_without_api_keys(make_settings):
    settings = make_settings(env="GLADIA_API_KEY=\nGROQ_API_KEY=\nGEMINI_API_KEY=\n")

    assert settings.gladia_api_key is None
    assert settings.groq_api_key is None
    assert settings.gemini_api_key is None


def test_api_key_from_env_file_is_hidden_in_repr(make_settings):
    settings = make_settings(env="GROQ_API_KEY=secret-value\n")

    assert settings.groq_api_key is not None
    assert settings.groq_api_key.get_secret_value() == "secret-value"
    assert "secret-value" not in repr(settings)


def test_unknown_key_in_config_section_is_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(toml='[log]\nlevle = "DEBUG"\n')


def test_unrelated_variable_in_env_file_is_ignored(make_settings):
    settings = make_settings(env="UV_LINK_MODE=copy\n")

    assert settings.log.level == "INFO"


def test_reads_llm_section_from_config_file(make_settings):
    settings = make_settings(toml=LLM_ANSWER.format(key="GROQ_API_KEY"))

    assert settings.llm.answer is not None
    assert settings.llm.answer.model == "openai/gpt-oss-120b"
    assert settings.llm.answer.params == {"reasoning_effort": "low"}
    assert settings.llm.summary is None


def test_unknown_llm_key_name_is_rejected_at_startup(make_settings):
    with pytest.raises(ValidationError, match="GROQ_KEY"):
        make_settings(toml=LLM_ANSWER.format(key="GROQ_KEY"))


def test_api_key_is_found_by_its_env_name(make_settings):
    settings = make_settings(env="GROQ_API_KEY=secret-value\n")

    key = settings.api_key("GROQ_API_KEY")

    assert key is not None
    assert key.get_secret_value() == "secret-value"
