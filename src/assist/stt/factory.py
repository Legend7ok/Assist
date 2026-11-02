from assist.settings import Settings
from assist.stt.base import Transcriber
from assist.stt.gladia import GladiaTranscriber


def create_transcriber(settings: Settings) -> Transcriber:
    """Build the speech recognition adapter named in the `[stt]` config section."""
    stt = settings.stt
    if stt is None:
        raise ValueError("[stt] is not configured in config/default.toml")

    api_key = settings.api_key(stt.api_key_env)
    if api_key is None:
        raise ValueError(f"{stt.api_key_env} is not set in .env or the environment")

    match stt.provider:
        case "gladia":
            return GladiaTranscriber(stt, api_key.get_secret_value())
