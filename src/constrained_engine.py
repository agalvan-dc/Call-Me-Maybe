"""Module for generating constrained JSON outputs from a small language model."""

import json
from pathlib import Path

import numpy as np

from llm_sdk import Small_LLM_Model
from src.tokenizer import FunctionDef, PromptDef


class ConstrainedEngine:
    """
    Engine for generating constrained JSON outputs from an LLM.

    This class handles token-by-token generation, applying logit masking
    to ensure the output adheres to a specified JSON structure and a
    restricted set of function names.
    """

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
            slm: The small language model instance used for generation.
            functions: A list of function definitions to constrain generation.
            prompts: A list of prompts to be processed.
            output_path: The file path where the generated output will be saved.
        """
        self.slm = slm
        self.functions = functions
        self.prompts = prompts
        self.output_path = output_path
        
        self.results: list[dict] = []
        self.valid_function_names = [fn.name for fn in self.functions]
        
        vocab_path = self.slm.get_path_to_vocab_file()
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab: dict = json.load(f)

        self.tti: dict[str, int] = {v: k for k, v in self.vocab.items()}
        self.open_brace_ids = [idx for tok, idx in self.tti.items() if "{" in tok]
        self.close_brace_ids = [idx for tok, idx in self.tti.items() if "}" in tok]
        self.colon_ids = [idx for tok, idx in self.tti.items() if ":" in tok]
        self.comma_ids = [idx for tok, idx in self.tti.items() if "," in tok]
        self.quote_ids = [idx for tok, idx in self.tti.items() if '"' in tok]

    def _build_system_prompt(self, prompt_text: str) -> str:
        """
        Construct the system prompt for the language model.

        Args:
            prompt_text: The user's input prompt text.

        Returns:
            The formatted system prompt containing function instructions.
        """
        sys_text = (
            "You are a helpful assistant with access to the following instructions\n"
        )
        fn_data = [fn.model_dump() for fn in self.functions]
        sys_text += json.dumps(fn_data, indent=4)
        sys_text += f"\n\nUser request: {prompt_text}\n"
        sys_text += "Generate a valid JSON with 'name', 'prompt' and 'parameters'"

        return sys_text

    def _is_generation_finished(
        self,
        curr_ids: list[int],
        prompt_len: int
    ) -> bool:
        """
        Determine if the token generation process is complete.

        Args:
            curr_ids: The current sequence of generated token IDs.
            prompt_len: The length of the original prompt in tokens.

        Returns:
            True if the generation should stop, False otherwise.
        """
        if len(curr_ids) - prompt_len > 512:
            return True
            
        generated_ids = curr_ids[prompt_len:]
        if not generated_ids:
            return False

        generated_text = self.slm.decode(generated_ids)
        if "{" not in generated_text:
            return False

        brace_count: int = 0
        for char in generated_text:
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
        return brace_count == 0 and "}" in generated_text

    def _is_field_closed(self, text: str, field_prefix: str) -> bool:
        """
        Check if a specific JSON field has been closed with a quote.

        Args:
            text: The generated text so far.
            field_prefix: The prefix of the field to check.

        Returns:
            True if the field is closed, False otherwise.
        """
        if field_prefix not in text:
            return False
        content_after_prefix = text.split(field_prefix)[-1]
        return '"' in content_after_prefix

    def _get_valid_next_tokens(
        self,
        curr_ids: list[int],
        prompt_len: int
    ) -> list[int]:
        """
        Identify valid token IDs that can be generated next.

        Args:
            curr_ids: The current sequence of generated token IDs.
            prompt_len: The length of the original prompt in tokens.

        Returns:
            A list of token IDs valid for the next generation step.
        """
        generated_text = self.slm.decode(curr_ids[prompt_len:])

        if not generated_text.strip():
            return self.open_brace_ids

        stripped = generated_text.strip()

        if stripped.endswith(('"name"', '"parameters"')):
            return self.colon_ids

        field_prefix = '"name": "'
        if field_prefix in stripped and not self._is_field_closed(stripped, field_prefix):
            curr_prefix = stripped.split(field_prefix)[-1]

            valid_ids = []
            for token_text, token_id in self.tti.items():
                candidate = curr_prefix + token_text
                if any(fn_name.startswith(candidate) for fn_name in self.valid_function_names):
                    valid_ids.append(token_id)
            return valid_ids

        if stripped.endswith(('"', '}', 'true')):
            return self.comma_ids + self.close_brace_ids

        return list(self.tti.values())

    def apply_logit_mask(self, logits: list[float], valid_ids: list[int]) -> list[float]:
        """
        Apply a mask to logits to prevent invalid tokens from being generated.

        Args:
            logits: The unnormalized predictions for the next token.
            valid_ids: The list of token IDs allowed to be generated.

        Returns:
            The masked logits as a list of floats.
        """
        logits_array = np.array(logits)
        masked_logits = np.full_like(logits_array, -np.inf)
        masked_logits[valid_ids] = logits_array[valid_ids]

        return masked_logits.tolist()

    def export_json(self) -> None:
        """
        Export the generated results to the output path as JSON.
        """
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4)

    def run(self) -> None:
        """
        Execute the constrained generation process for all prompts.
        """
        for prompt_def in self.prompts:
            context_text = self._build_system_prompt(prompt_def.prompt)
            curr_token_ids: list[int] = self.slm.encode(context_text)
            prompt_len = len(curr_token_ids)

            while not self._is_generation_finished(curr_token_ids, prompt_len):
                logits = self.slm.get_logits_from_input_ids(curr_token_ids)
                valid_ids = self._get_valid_next_tokens(curr_token_ids, prompt_len)
                masked_logits = self.apply_logit_mask(logits, valid_ids)

                next_token_id = int(np.argmax(masked_logits))
                curr_token_ids.append(next_token_id)
                
            generated_text = self.slm.decode(curr_token_ids[prompt_len:])
            self.results.append({
                "prompt": prompt_def.prompt,
                "generated": generated_text
            })
            
        self.export_json()
