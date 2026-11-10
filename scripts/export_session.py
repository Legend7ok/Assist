"""Export a session file to Markdown again, for example after the app crashed.

Usage: uv run python scripts/export_session.py sessions/2026-10-28_14-05-33.jsonl
The Markdown file is written next to it, with the same name and the .md extension.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from assist.logging_setup import configure_logging
from assist.settings import Settings
from assist.storage.export import export_session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="session .jsonl file")
    args = parser.parse_args()
    configure_logging(Settings().log.level)

    try:
        markdown = export_session(args.session)
    except FileNotFoundError:
        sys.exit(f"No such file: {args.session}")
    except ValidationError as error:
        # A broken line in the middle of the file, not a cut-off last one.
        sys.exit(f"{args.session} is damaged: {error}")
    except ValueError as error:
        sys.exit(f"{args.session}: {error}")
    print(markdown)


if __name__ == "__main__":
    main()
