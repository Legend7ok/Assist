"""Application settings.

Sources by priority: environment variables, then `.env`, then `config/default.toml`,
then field defaults. API keys are read from `.env` / the environment only and are
never stored in the config file.
"""

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
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


class LLMEndpoint(Section):
    provider: Literal["openai_compat"]
    base_url: str
    model: str
    api_key_env: str
    params: dict[str, Any] = Field(default_factory=dict)
    timeout: float = Field(default=30.0, gt=0)


class LLMSettings(Section):
    # A section left out of the config is None: the app still starts, and only the part that
    # needs this model fails, with a message naming the missing section.
    answer: LLMEndpoint | None = None
    summary: LLMEndpoint | None = None


class ContextSettings(Section):
    window_seconds: float = Field(default=300.0, gt=0)
    compress_seconds: float = Field(default=120.0, gt=0)

    @model_validator(mode="after")
    def _check_compress_part(self) -> Self:
        if self.compress_seconds >= self.window_seconds:
            raise ValueError("context.compress_seconds must be less than window_seconds")
        return self


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
    llm: LLMSettings = LLMSettings()
    context: ContextSettings = ContextSettings()

    # Explicit aliases drop the ASSIST_ prefix so keys keep the names providers use.
    # Optional so the app starts before the stages that need them are reached.
    gladia_api_key: SecretStr | None = Field(default=None, validation_alias="GLADIA_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    gemini_api_key: SecretStr | None = Field(default=None, validation_alias="GEMINI_API_KEY")

    # Keys are looked up here and not in os.environ: pydantic-settings reads .env itself and
    # does not export it, so a key kept only in .env is invisible to os.environ.
    def api_key(self, env_name: str) -> SecretStr | None:
        return getattr(self, self._api_key_fields()[env_name])

    @model_validator(mode="after")
    def _check_llm_key_names(self) -> Self:
        known = self._api_key_fields()
        for purpose in LLMSettings.model_fields:
            endpoint: LLMEndpoint | None = getattr(self.llm, purpose)
            if endpoint is not None and endpoint.api_key_env not in known:
                raise ValueError(
                    f"llm.{purpose}.api_key_env: unknown key {endpoint.api_key_env!r}, "
                    f"expected one of {sorted(known)}"
                )
        return self

    @classmethod
    def _api_key_fields(cls) -> dict[str, str]:
        return {
            field.validation_alias: name
            for name, field in cls.model_fields.items()
            if isinstance(field.validation_alias, str)
        }

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
