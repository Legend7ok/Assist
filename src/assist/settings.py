"""Application settings.

Sources by priority: environment variables, then `.env`, then `config/default.toml`,
then field defaults. API keys are read from `.env` / the environment only and are
never stored in the config file.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# src/assist/settings.py -> project root. Anchoring paths here keeps the app working
# regardless of the directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = PROJECT_ROOT / "config" / "default.toml"
ENV_FILE = PROJECT_ROOT / ".env"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Section(BaseModel):
    """Base for config sections: a mistyped key inside a section fails loudly."""

    model_config = ConfigDict(extra="forbid")


class LogSettings(Section):
    level: LogLevel = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        toml_file=CONFIG_FILE,
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="ASSIST_",
        env_nested_delimiter="__",
        # An empty `KEY=` line copied from .env.example means "not set", not an empty key.
        env_ignore_empty=True,
        # .env may hold variables for other tools (uv, IDE), so unknown top-level keys are
        # skipped. Typos inside config sections are still rejected by Section.
        extra="ignore",
    )

    log: LogSettings = LogSettings()

    # Explicit aliases drop the ASSIST_ prefix so keys keep the names providers use.
    # Optional so the app starts before the stages that need them are reached.
    gladia_api_key: SecretStr | None = Field(default=None, validation_alias="GLADIA_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
        )
