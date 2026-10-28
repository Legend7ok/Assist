from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from assist.settings import Settings

ENV_VARS = (
    "ASSIST_LOG__LEVEL",
    "ASSIST_AUDIO__LOOPBACK_DEVICE",
    "ASSIST_AUDIO__MIC_DEVICE",
    "ASSIST_AUDIO__FRAME_MS",
    "GLADIA_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
)


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


def test_audio_devices_default_to_the_windows_ones(make_settings):
    settings = make_settings()

    assert settings.audio.loopback_device is None
    assert settings.audio.mic_device is None


def test_frame_length_from_environment_variable_is_accepted(make_settings, monkeypatch):
    monkeypatch.setenv("ASSIST_AUDIO__FRAME_MS", "30")

    settings = make_settings()

    assert settings.audio.frame_ms == 30


def test_frame_length_webrtcvad_cannot_take_is_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(toml="[audio]\nframe_ms = 25\n")


def test_vad_aggressiveness_outside_webrtcvad_modes_is_rejected(make_settings):
    with pytest.raises(ValidationError):
        make_settings(toml="[audio]\nvad_aggressiveness = 4\n")
