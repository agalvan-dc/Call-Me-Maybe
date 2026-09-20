"""Constrained execution engine for LLM function calling."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from llm_sdk import Small_LLM_Model

from src.tokenizer import FunctionDef, PromptDef


class ConstrainedEngine:
    """Constrained decoding engine that forces schema-compliant JSON output."""

    def __init__(
        self,
        slm: Small_LLM_Model,
        functions: list[FunctionDef],
        prompts: list[PromptDef],
        output_path: Path,
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
        self._fn_names: list[str] = [fn.name for fn in functions]
        self._vocab_size = 0
        self._tok_text: list[str] = []
        self._ctrl_ids: set[int] = set()
        self._quote_mid_ids: set[int] = set()
        self._quote_end_ids: set[int] = set()
        self._backslash_ids: set[int] = set()
        self._escape_start_ids: set[int] = set()

    def _encode(self, text: str) -> list[int]:
        """Encode a fixed structural fragment into a flat list of token ids."""
        encoded: Any = self.slm.encode(text)
        if hasattr(encoded, "tolist"):
            flat = encoded.tolist()
        else:
            flat = list(encoded)
        if flat and isinstance(flat[0], list):
            flat = flat[0]
        return [int(x) for x in flat]

    def _pick(self, logits: list[float], excluded: list[int]) -> int:
        """Return the argmax token id after masking the excluded candidates."""
        arr = np.asarray(logits, dtype=np.float32)
        if excluded:
            arr[list(excluded)] = -np.inf
        return int(np.argmax(arr))

    def _ensure_cache(self, vocab_size: int) -> None:
        """Build the id-to-text cache and the string safety flag sets."""
        if self._tok_text:
            return
        self._vocab_size = vocab_size
        ctrl: set[int] = set()
        quote_mid: set[int] = set()
        quote_end: set[int] = set()
        backslash: set[int] = set()
        escape_start: set[int] = set()
        texts: list[str] = []
        for i in range(vocab_size):
            piece = self.slm.decode([i])
            texts.append(piece)
            if any(ord(ch) < 0x20 for ch in piece):
                ctrl.add(i)
            qpos = piece.find('"')
            if qpos != -1:
                if qpos == len(piece) - 1:
                    quote_end.add(i)
                else:
                    quote_mid.add(i)
            if "\\" in piece:
                backslash.add(i)
            if piece and piece[0] in '"\\/bfnrtu':
                escape_start.add(i)
        self._tok_text = texts
        self._ctrl_ids = ctrl
        self._quote_mid_ids = quote_mid
        self._quote_end_ids = quote_end
        self._backslash_ids = backslash
        self._escape_start_ids = escape_start

    @staticmethod
    def _num_status(value: str) -> str:
        """Classify a JSON number as complete, partial, or invalid."""
        if not value:
            return "partial"
        text = value
        i = 0
        n = len(text)
        if text[i] == "-":
            i += 1
            if i == n:
                return "partial"
        if text[i] == "0":
            i += 1
        elif "1" <= text[i] <= "9":
            i += 1
            while i < n and text[i].isdigit():
                i += 1
        else:
            return "invalid"
        if i == n:
            return "complete"
        if text[i] == ".":
            i += 1
            if i == n:
                return "partial"
            if not text[i].isdigit():
                return "invalid"
            while i < n and text[i].isdigit():
                i += 1
            if i == n:
                return "complete"
        if text[i] in "eE":
            i += 1
            if i == n:
                return "partial"
            if text[i] in "+-":
                i += 1
                if i == n:
                    return "partial"
            while i < n and text[i].isdigit():
                i += 1
            if i == n:
                return "complete"
            return "invalid"
        return "invalid"

    @staticmethod
    def _string_token(piece: str, escaped: bool) -> tuple[bool, bool, bool]:
        """Validate a token as JSON string content.

        Returns a tuple ``(ok, escaped_at_end, terminated)`` describing how the
        token interacts with the JSON string grammar.
        """
        for idx, ch in enumerate(piece):
            if escaped:
                if ch in '"\\/bfnrtu':
                    escaped = False
                else:
                    return False, False, False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                if idx == len(piece) - 1:
                    return True, False, True
                return False, False, False
            elif ord(ch) < 0x20:
                return False, False, False
        return True, escaped, False

    def _gen_name(self, input_ids: list[int]) -> tuple[str, list[int]]:
        """Generate a function name constrained to the known function list."""
        targets = self._fn_names
        logits = self.slm.get_logits_from_input_ids(input_ids)
        self._ensure_cache(len(logits))
        name = ""
        for _ in range(64):
            alive = [t for t in targets if t.startswith(name)]
            if not alive:
                break
            excluded: list[int] = []
            for i, piece in enumerate(self._tok_text):
                if piece == "" or not any(
                        t.startswith(name + piece) for t in alive):
                    excluded.append(i)
            chosen = self._pick(logits, excluded)
            piece = self._tok_text[chosen]
            input_ids.append(chosen)
            name += piece
            if name in targets and not any(
                    t for t in targets
                    if t != name and t.startswith(name)):
                break
            logits = self.slm.get_logits_from_input_ids(input_ids)
        if name not in targets:
            alive = [t for t in targets if t.startswith(name)]
            fallback = min(alive, key=len) if alive else targets[0]
            tail = fallback[len(name):]
            if tail:
                input_ids.extend(self._encode(tail))
            name = fallback
        return name, input_ids

    def _gen_number(self, input_ids: list[int]) -> tuple[str, list[int]]:
        """Generate a number value constrained to the JSON number grammar."""
        acc = ""
        for _ in range(40):
            logits = self.slm.get_logits_from_input_ids(input_ids)
            top = int(np.argmax(logits))
            top_text = self._tok_text[top].strip()
            if acc and self._num_status(acc) == "complete":
                if not top_text or self._num_status(
                        acc + top_text) == "invalid":
                    return acc, input_ids
            excluded: list[int] = []
            for i, piece in enumerate(self._tok_text):
                stripped = piece.strip()
                if stripped and self._num_status(acc + stripped) == "invalid":
                    excluded.append(i)
            chosen = self._pick(logits, excluded)
            chosen_text = self._tok_text[chosen].strip()
            input_ids.append(chosen)
            if chosen_text:
                acc += chosen_text
            if acc and self._num_status(acc) == "complete":
                return acc, input_ids
        if acc and self._num_status(acc) != "invalid":
            return acc, input_ids
        return "0", input_ids

    def _gen_bool(self, input_ids: list[int]) -> tuple[str, list[int]]:
        """Generate a boolean constrained to the literals true or false."""
        targets = ("true", "false")
        acc = ""
        for _ in range(12):
            logits = self.slm.get_logits_from_input_ids(input_ids)
            excluded: list[int] = []
            for i, piece in enumerate(self._tok_text):
                stripped = piece.strip()
                if stripped and not any(
                        t.startswith(acc + stripped) for t in targets):
                    excluded.append(i)
            chosen = self._pick(logits, excluded)
            chosen_text = self._tok_text[chosen].strip()
            input_ids.append(chosen)
            if chosen_text:
                acc += chosen_text
            if acc in targets:
                return acc, input_ids
        alive = [t for t in targets if t.startswith(acc)]
        fallback = alive[0] if alive else "false"
        tail = fallback[len(acc):]
        if tail:
            input_ids.extend(self._encode(tail))
        return fallback, input_ids

    def _gen_string(self, input_ids: list[int]) -> tuple[str, list[int]]:
        """Generate a string value constrained to valid JSON string content."""
        acc = ""
        escaped = False
        for _ in range(48):
            logits = self.slm.get_logits_from_input_ids(input_ids)
            if escaped:
                allowed = [
                    i for i in self._escape_start_ids
                    if self._string_token(self._tok_text[i], True)[0]
                ]
                excluded = set(range(self._vocab_size)) - set(allowed)
            else:
                excluded = set(self._ctrl_ids)
                excluded |= self._quote_mid_ids
                for i in self._quote_end_ids | self._backslash_ids:
                    if not self._string_token(self._tok_text[i], False)[0]:
                        excluded.add(i)
            chosen = self._pick(logits, list(excluded))
            piece = self._tok_text[chosen]
            ok, new_escaped, terminated = self._string_token(piece, escaped)
            input_ids.append(chosen)
            if not ok:
                break
            escaped = new_escaped
            if terminated:
                acc += piece[:-1]
                return acc, input_ids
            acc += piece
        if escaped:
            input_ids.extend(self._encode("\\\\"))
        input_ids.extend(self._encode('"'))
        return acc, input_ids

    def _coerce_value(self, value: Any, ptype: str) -> Any:
        """Cast a raw generated value to the schema-declared Python type."""
        ptype = ptype.lower()
        if ptype in ("number", "float"):
            text = str(value)
            try:
                if "." in text or "e" in text.lower():
                    return float(text)
                return int(text)
            except ValueError:
                return value
        if ptype == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                return value
        if ptype in ("boolean", "bool"):
            return str(value).strip().lower() == "true"
        return str(value) if value is not None else ""

    def _build_system_prompt(self, prompt_text: str) -> str:
        """Build a compressed system prompt with the available functions."""
        lines = ["AVAILABLE FUNCTIONS:"]
        for fn in self.functions:
            props = self._function_props(fn)
            param_names = list(props.keys())
            lines.append(f"{fn.name}({', '.join(param_names)}) "
                         f"- {fn.description}")

        examples: list[tuple[str, str, dict[str, Any], list[str]]] = [
            ("fn_add_numbers", "What is the sum of 2 and 3?",
             {"a": 2, "b": 3}, ["a", "b"]),
            ("fn_greet", "Greet Alice", {"name": "Alice"}, ["name"]),
            ("fn_reverse_string", "Reverse the string 'hi'",
             {"s": "hi"}, ["s"]),
            ("fn_get_square_root", "What is the square root of 16?",
             {"a": 16}, ["a"]),
            ("fn_substitute_string_with_regex",
             "Replace all numbers in 'a1' with X",
             {"source_string": "a1", "regex": "\\d", "replacement": "X"},
             ["source_string", "regex", "replacement"]),
        ]
        for fn_name, user, params, keys in examples:
            target = next(
                (f for f in self.functions if f.name == fn_name), None)
            if target is None:
                continue
            props = self._function_props(target)
            if list(props.keys()) != keys:
                continue
            payload = json.dumps({"name": fn_name, "parameters": params},
                                 separators=(",", ":"), ensure_ascii=False)
            lines.append(f"User: {user}")
            lines.append(f"Assistant: {payload}")

        lines.append(f"User: {prompt_text}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _parse_slice(self, text: str) -> dict[str, Any]:
        """Parse a decoded JSON string, returning an empty dict on failure."""
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return {}

    def export_json(self) -> None:
        """Export generated function calls to the output JSON file."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=4)

    def _function_props(self, fn: FunctionDef) -> dict[str, dict[str, str]]:
        """Return the parameter property map of a function definition."""
        if "properties" in fn.parameters:
            return cast(dict[str, dict[str, str]],
                        fn.parameters["properties"])
        return fn.parameters

    def run(self) -> None:
        """Run the constrained generation loop for every prompt."""
        fn_by_name = {fn.name: fn for fn in self.functions}
        for prompt_def in self.prompts:
            context = self._build_system_prompt(prompt_def.prompt)
            input_ids = self._encode(context)
            input_ids.extend(self._encode('{"name":"'))

            gen_start = len(input_ids)
            name, input_ids = self._gen_name(input_ids)

            fn = fn_by_name.get(name)
            if fn is None:
                fn = self.functions[0]
            props = self._function_props(fn)
            keys = list(props.keys())

            input_ids.extend(self._encode('","parameters":{'))
            params: dict[str, Any] = {}
            for idx, key in enumerate(keys):
                input_ids.extend(self._encode(json.dumps(key) + ":"))
                ptype = props[key].get("type", "string").lower()
                if ptype in ("number", "float", "integer"):
                    raw, input_ids = self._gen_number(input_ids)
                elif ptype in ("boolean", "bool"):
                    raw, input_ids = self._gen_bool(input_ids)
                else:
                    raw, input_ids = self._gen_string(input_ids)
                params[key] = self._coerce_value(raw, ptype)
                if idx != len(keys) - 1:
                    input_ids.extend(self._encode(","))
            input_ids.extend(self._encode("}}"))

            json_text = self.slm.decode(input_ids[gen_start:])
            parsed = self._parse_slice(json_text)
            if parsed:
                params = {
                    k: self._coerce_value(
                        v, props.get(k, {}).get("type", "string"))
                    for k, v in parsed.get("parameters", {}).items()
                }

            self.results.append({
                "prompt": prompt_def.prompt,
                "name": name,
                "parameters": params,
            })

        self.export_json()
