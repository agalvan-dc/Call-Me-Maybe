#!/usr/bin/env python3

import sys
import time
from json import JSONDecodeError

from llm_sdk import Small_LLM_Model  # type: ignore
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
        parsed_data = parser.parse_and_load()
        if parsed_data is None:
            raise ValueError("Something went wrong in parsing....")
        functions, prompts, output_path = parsed_data
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
        start_time = time.perf_counter()
        engine.run()
    except Exception as e:
        print(f"\033[1;31mEngine Error - {e} \033[0m")
        sys.exit(1)

    elapsed_time = time.perf_counter() - start_time
    print("\033[1;32mExecution completed in "
          f"{elapsed_time:.2f} seconds.\033[0m")


if __name__ == "__main__":
    main()
