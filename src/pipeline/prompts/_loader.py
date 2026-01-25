"""
Load prompt templates from .md and _schema.txt files.

Used by hops when building prompts. Finance experts edit the files in
`templates/`; this module just reads and fills placeholders.
"""

from __future__ import annotations

from pathlib import Path

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def load_template(name: str) -> str:
    """Load prompt markdown for a hop (e.g. entity_grounding.md)."""
    path = _TEMPLATES_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def load_schema(name: str) -> str:
    """Load JSON schema text for a hop (e.g. entity_grounding_schema.txt)."""
    path = _TEMPLATES_DIR / f"{name}_schema.txt"
    return path.read_text(encoding="utf-8").strip()


def build_from_template(name: str, **kwargs: str) -> str:
    """
    Load template + schema, fill placeholders, return final prompt.

    kwargs must include all placeholders used in the .md except json_schema,
    which is loaded from {name}_schema.txt if not provided.
    """
    template = load_template(name)
    if "json_schema" not in kwargs:
        kwargs = {**kwargs, "json_schema": load_schema(name)}
    out = template.format(**kwargs)
    return out.replace("\n\n\n", "\n\n").strip()
