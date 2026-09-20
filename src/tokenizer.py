"""Pydantic schemas for data validation and JSON loading utilities."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class FunctionDef(BaseModel):
    """Pydantic model to validate function definitions."""

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class PromptDef(BaseModel):
    """Pydantic model to validate input prompts."""

    prompt: str


def load_and_validate_json(filepath: Path,
                           model: type[T]) -> list[T]:
    """
    Load a JSON file and validate its content against a Pydantic model.

    Catches file reading and validation exceptions to prevent crashes.

    Args:
        filepath: The path to the JSON file to be read.
        model: The Pydantic model class to validate the data against.

    Returns:
        A list of validated Pydantic model instances. Returns an empty
        list if a file or validation error occurs.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        return [model(**item) for item in raw_data]

    except FileNotFoundError:
        print(f"Error: File {filepath} does not exist.")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: JSON at {filepath} is malformed - {e}")
        return []
    except ValidationError as e:
        print(f"Pydantic validation error at {filepath} - {e}")
        return []
