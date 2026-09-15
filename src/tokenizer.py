import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


class FunctionDef(BaseModel):
    """Pydantic model to validate function definitions."""
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class PromptDef(BaseModel):
    """Pydantic model to validate input prompts."""
    prompt: str


def load_and_validate_json[T: BaseModel](filepath: Path,
                                         model: type[T]) -> list[T]:
    """
    Loads a JSON file and validates its content against a Pydantic model.
    Catches file reading and validation exceptions to prevent crashes.
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
