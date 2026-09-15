"""Command Line Interface argument parser and JSON loader."""

import argparse
from pathlib import Path

from .tokenizer import FunctionDef, PromptDef
from .tokenizer import load_and_validate_json as lvj


class Parser:
    """Manage CLI arguments and handle the validated loading of JSON files."""

    def __init__(self) -> None:
        """Initialize the Parser with default argument definitions."""
        self.parser = argparse.ArgumentParser(description="Function "
                                              "Calling LLM Tool")
        self.parser.add_argument(
            "--functions_definition",
            type=Path,
            default=Path("data/input/functions_definition.json")
        )
        self.parser.add_argument(
            "--input",
            type=Path,
            default=Path("data/input/function_calling_tests.json")
        )
        self.parser.add_argument(
            "--output",
            type=Path,
            default=Path("data/output/function_calling_results.json")
        )

    def parse_and_load(self) -> tuple[list[FunctionDef],
                                      list[PromptDef], Path] | None:
        """
        Parse arguments and return the validated data ready for processing.

        Returns:
            A tuple containing a list of valid function definitions, a list of 
            valid prompts, and the desired output path.

        Raises:
            ValueError: If critical data (functions or prompts) fails to load.
        """
        args = self.parser.parse_args()

        functions = lvj(args.functions_definition, FunctionDef)
        prompts = lvj(args.input, PromptDef)

        if not functions or not prompts:
            raise ValueError("Critical error loading data.")

        return functions, prompts, args.output
