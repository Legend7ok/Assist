import pytest

from assist.stt.factory import create_transcriber
from assist.stt.gladia import GladiaTranscriber

STT = """
[stt]
provider = "gladia"
api_key_env = "GLADIA_API_KEY"
"""


async def test_builds_the_adapter_named_in_config(make_settings):
    settings = make_settings(toml=STT, env="GLADIA_API_KEY=secret-value\n")

    transcriber = create_transcriber(settings)

    assert isinstance(transcriber, GladiaTranscriber)
    await transcriber.close()


def test_missing_api_key_is_named_in_the_error(make_settings):
    settings = make_settings(toml=STT)

    with pytest.raises(ValueError, match="GLADIA_API_KEY"):
        create_transcriber(settings)


def test_section_missing_from_config_is_named_in_the_error(make_settings):
    settings = make_settings(env="GLADIA_API_KEY=secret-value\n")

    with pytest.raises(ValueError, match=r"\[stt\]"):
        create_transcriber(settings)
