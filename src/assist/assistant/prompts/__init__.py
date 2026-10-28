from importlib.resources import files


def load_prompt(name: str) -> str:
    """Text of `<name>.md` from this folder, shipped inside the package."""
    return files(__package__).joinpath(f"{name}.md").read_text(encoding="utf-8")
