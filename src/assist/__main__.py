import structlog

from assist.logging_setup import configure_logging
from assist.settings import Settings


def main() -> None:
    settings = Settings()
    configure_logging(settings.log.level)
    # Not __name__: under `python -m assist` it is "__main__", under the `assist` command
    # it is "assist.__main__". A fixed name keeps the startup line the same either way.
    structlog.get_logger("assist").info("assist_started", log_level=settings.log.level)


if __name__ == "__main__":
    main()
