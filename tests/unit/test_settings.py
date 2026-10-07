from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from assist.settings import Settings

ENV_VARS = ("ASSIST_LOG__LEVEL", "GLADIA_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")


@pytest.fixture
def make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build Settings from temporary config and .env files, isolated from the real ones."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    def factory(toml: str = "", env: str = "") -> Settings:
        toml_file = tmp_path / "default.toml"
        env_file = tmp_path / ".env"
        toml_file.write_text(toml, encoding="utf-8")
        env_file.write_text(env, encoding="utf-8")

        class IsolatedSettings(Settings):
            model_config = SettingsConfigDict(toml_file=toml_file, env_file=env_file)

        return IsolatedSettings()

    return factory


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
