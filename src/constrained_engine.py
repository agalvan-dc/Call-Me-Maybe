"""Constrained execution engine for LLM function calling."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from llm_sdk import Small_LLM_Model  # type: ignore
from src.tokenizer import FunctionDef, PromptDef


class ConstrainedEngine:
    """Engine for generating constrained JSON outputs from an LLM."""

    def __init__(
        self,
        slm: Small_LLM_Model,
        functions: list[FunctionDef],
        prompts: list[PromptDef],
        output_path: Path
    ) -> None:
        """
        Initialize the ConstrainedEngine.

        Args:
            slm: The initialized language model instance.
            functions: A list of parsed function definitions.
            prompts: A list of prompts to process.
            output_path: The file path where the generated JSON will be saved.
        """
        self.slm = slm
        self.functions = functions
        self.prompts = prompts
        self.output_path = output_path
        self.results: list[dict[str, Any]] = []

    def _build_system_prompt(self, prompt_text: str) -> str:
        """
        Build an ultra-compressed system prompt.

        Uses a single master example for zero-shot precision.

        Args:
            prompt_text: The user's input prompt to be processed.

        Returns:
            The fully formatted system prompt string.
        """
        fn_lines: list[str] = []
        for fn in self.functions:
            props = (fn.parameters.get("properties", fn.parameters)
                     if isinstance(fn.parameters, dict) else {})
            param_names = list(props.keys())
            fn_lines.append(f"{fn.name}({', '.join(param_names)})")

        schema_str = " | ".join(fn_lines)

        ex = '{"name":"fn_substitute_string_with_regex"'
        ex += ',"parameters":{"source_string":"a1"'
        ex += ',"regex":"\\d","replacement":"X"}}'

        return (
            f"S:{schema_str}\n"
            f"Q:replace numbers in 'a1'\nA:{ex}\n"
            f"Q:{prompt_text}\nA:{{\"name\":\""
        )

    def _parse_generated_json(self, text: str) -> tuple[str, dict[str, Any]]:
        """
        Parse generated text into a function name and parameter dictionary.

        Args:
            text: The raw generated text from the language model.

        Returns:
            A tuple containing the function name (str) 
                                  and its parameters (dict).
            Returns an empty string and dictionary if parsing fails.
        """
        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data: dict[str, Any] = json.loads(match.group(0))
                if isinstance(data, dict):
                    name = str(data.get("name", ""))
                    params = data.get("parameters", {})
                    if isinstance(params, dict):
                        return name, params
        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
            pass
        return "", {}

    def _coerce_types(self, func_name: str,
                      params: dict[str, Any]) -> dict[str, Any]:
        """
        Coerce parameter data types against the schema definition.

        Args:
            func_name: The name of the function to check against the schema.
            params: The raw dictionary of parameters extracted from the JSON.

        Returns:
            A new dictionary of parameters with properly cast data types.
        """
        target_fn = next((fn for fn in self.functions
                          if fn.name == func_name), None)

        if not target_fn or not isinstance(target_fn.parameters, dict):
            return params

        fn_params: dict[str, Any] = target_fn.parameters
        schema_props = fn_params.get("properties", fn_params)
        coerced: dict[str, Any] = {}

        for key, val in params.items():
            if key not in schema_props:
                coerced[key] = val
                continue

            expected_type = (str(schema_props[key].get("type", "")).lower()
                             if isinstance(schema_props[key], dict) else "")

            try:
                if expected_type in ("number", "float"):
                    coerced[key] = float(val) if "." in str(val) else int(val)
                elif expected_type == "integer":
                    coerced[key] = int(val)
                elif expected_type == "string":
                    coerced[key] = str(val)
                elif expected_type == "boolean":
                    coerced[key] = str(val).lower() in ("true", "1")
                else:
                    coerced[key] = val
            except (ValueError, TypeError):
                coerced[key] = val

        return coerced

    def export_json(self) -> None:
        """
        Export generated function calls to the output JSON file.
        
        Creates the necessary parent directories if they do not exist.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)

    def run(self) -> None:
        """
        Execute a high-speed generation loop.

        Uses deferred parsing to eliminate inner-loop latency.
        """
        for prompt_def in self.prompts:
            context = self._build_system_prompt(prompt_def.prompt)
            encoded: Any = self.slm.encode(context)

            curr_ids: list[int] = (encoded.tolist() if
                                   hasattr(encoded, "tolist")
                                   else list(encoded))
            if curr_ids and isinstance(curr_ids[0], list):
                curr_ids = curr_ids[0]

            generated = '{"name":"'
            open_braces = 1

            for _ in range(38):
                logits: Any = self.slm.get_logits_from_input_ids(curr_ids)
                next_id = (int(logits.argmax() if hasattr(logits, "argmax")
                               else np.argmax(logits)))

                curr_ids.append(next_id)
                new_str = str(self.slm.decode([next_id]))
                generated += new_str

                open_braces += new_str.count("{") - new_str.count("}")

                if open_braces <= 0:
                    break

            func_name, raw_params = self._parse_generated_json(generated)
            coerced_params = self._coerce_types(func_name, raw_params)

            self.results.append({
                "prompt": prompt_def.prompt,
                "name": func_name,
                "parameters": coerced_params
            })

        self.export_json()
