from pathlib import Path

import pytest
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
