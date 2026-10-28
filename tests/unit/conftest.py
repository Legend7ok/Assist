import os
from pathlib import Path

import pytest
from pydantic_settings import SettingsConfigDict

from assist.settings import Settings

API_KEYS = ("GLADIA_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")


@pytest.fixture
def make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build Settings from temporary config and .env files, isolated from the real ones."""
    # Any ASSIST_ override left in the terminal (for example a shorter window for a manual
    # run) would otherwise leak into the settings under test.
    overrides = [name for name in os.environ if name.upper().startswith("ASSIST_")]
    for name in (*API_KEYS, *overrides):
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
