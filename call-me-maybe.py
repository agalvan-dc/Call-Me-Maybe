#!/usr/bin/env python3
"""Main entry point for the constrained function calling engine."""

import sys
from json import JSONDecodeError

from llm_sdk import Small_LLM_Model
from src import ConstrainedEngine, Parser


def main() -> None:
    """
    Execute the parsing, initialization, and run phases of the engine.

    Parses input constraints and prompts, initializes the language model
    instance, and runs the constrained generation pipeline. Gracefully
    handles exceptions by printing error messages and exiting safely.
    """
    try:
        parser = Parser()
        functions, prompts, output_path = parser.parse_and_load()
    except (OSError, ValueError, JSONDecodeError) as e:
        print(f"\033[1;31mError - {e} \033[0m")
        sys.exit(1)

    try:
        slm_instance = Small_LLM_Model()
        engine = ConstrainedEngine(
            slm=slm_instance,
            functions=functions,
            prompts=prompts,
            output_path=output_path
        )
        engine.run()
    except Exception as e:
        print(f"\033[1;31mEngine Error - {e} \033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
