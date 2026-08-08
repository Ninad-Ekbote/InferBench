"""Shared, side-effect-light helpers: logging setup and prompt handling.

Kept free of any vLLM/HTTP specifics so it can be reused by unrelated
tooling (e.g. a future results-analysis script) without pulling in the
networking stack.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure root logging once and return the benchmark's named logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("benchmark")


def load_prompts(path: Path) -> list[str]:
    """Load prompts from a JSON file.

    Accepts either a flat list of strings, or a list of objects with a
    "prompt" key, so callers can attach extra metadata per prompt later
    without breaking this loader:

        ["prompt one", "prompt two"]
        [{"prompt": "prompt one", "category": "qa"}, ...]

    Raises:
        FileNotFoundError: if `path` does not exist.
        ValueError: if the file is empty or malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Prompts file {path} must contain a non-empty JSON list")

    prompts: list[str] = []
    for item in data:
        if isinstance(item, str):
            prompts.append(item)
        elif isinstance(item, dict) and "prompt" in item:
            prompts.append(str(item["prompt"]))
        else:
            raise ValueError(f"Unsupported prompt entry in {path}: {item!r}")
    return prompts


def adjust_prompt_length(prompt: str, target_words: int | None) -> str:
    """Adjust a prompt to approximately `target_words` words.

    Truncates prompts longer than the target and pads shorter ones by
    repeating their own text, so prompt length can be varied as an
    independent benchmark parameter regardless of what's in prompts.json.
    A `target_words` of None leaves the prompt untouched.
    """
    if target_words is None:
        return prompt

    words = prompt.split()
    if not words:
        return prompt

    if len(words) >= target_words:
        return " ".join(words[:target_words])

    repeats = target_words // len(words) + 1
    return " ".join((words * repeats)[:target_words])
