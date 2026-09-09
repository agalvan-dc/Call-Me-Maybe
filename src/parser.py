import argparse
from pathlib import Path

from .tokenizer import FunctionDef, PromptDef
from .tokenizer import load_and_validate_json as lvj


class Parser:
    """Manages CLI arguments and handles the validated loading of JSON files."""
    
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(description="Function Calling LLM Tool")
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

    def parse_and_load(self) -> tuple[list[FunctionDef], list[PromptDef], Path] | None:
        """Parses arguments and returns the validated data ready for processing."""
        args = self.parser.parse_args()
        
        functions = lvj(args.functions_definition, FunctionDef)
        prompts = lvj(args.input, PromptDef)
        
        if not functions or not prompts:
            raise ValueError("Critical error loading data.")
            
        return functions, prompts, args.output
