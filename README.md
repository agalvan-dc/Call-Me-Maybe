*This project has been created as part of the 42 curriculum by agalvan-.*

# Call Me Maybe - Constrained Function Calling Engine

## Description
This project implements a constrained function calling engine designed to translate natural language prompts into structured, machine-executable JSON function calls. Large Language Models (LLMs) are powerful at understanding text, but small models (like the 0.6B parameter model used here) often struggle to produce reliable, properly formatted JSON output. 

The goal of this project is to bridge that gap. By utilizing constrained decoding techniques, the system intervenes in the text generation process token-by-token. It ensures that the output is not only 100% syntactically valid JSON but also strictly adheres to predefined function schemas (correct function names, accurate argument types, and all required keys). This transforms a lightweight language model into a highly reliable structured data extractor and function dispatcher.

## Algorithm Explanation
The core of this engine relies on **Constrained Decoding**. A traditional LLM generates text by predicting a probability distribution (logits) for the next token and selecting the most likely one. Relying purely on prompting for structured data is highly error-prone.

Our algorithm manipulates this generation process directly:
1. **Schema Parsing:** The system first reads the `functions_definition.json` using Pydantic models to understand exactly what structures, keys, and data types are permitted.
2. **Vocabulary Mapping:** It utilizes the provided `llm_sdk` to access the model's vocabulary, mapping token IDs to their exact string representations (including spaces and special characters).
3. **Logit Masking:** At every single generation step, the engine evaluates the current state of the JSON being built. It identifies which tokens would maintain a valid JSON syntax and comply with the expected function schema.
4. **Token Filtering:** The logits for all invalid tokens are forcefully set to negative infinity (`-inf`).
5. **Generation:** The model is then forced to sample only from the remaining valid tokens. This loop repeats until the JSON object is completely generated, guaranteeing 100% structural and semantic compliance without relying on the LLM's spontaneous formatting capabilities.

## Architecture and Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI as Parser (CLI)
    participant Engine as ConstrainedEngine
    participant SDK as Small_LLM_Model
    participant Output as JSON File

    User->>CLI: uv run python -m src
    CLI->>CLI: Validate JSON schemas (Pydantic)
    CLI->>Engine: Initialize with Prompts & Functions
    Engine->>SDK: Build System Prompt
    
    loop Token-by-Token Generation
        Engine->>SDK: get_logits_from_input_ids(current_tokens)
        SDK-->>Engine: Logits distribution
        Engine->>Engine: Mask invalid tokens (-inf)
        Engine->>Engine: Select valid token
        Engine->>Engine: Append to current output
    end
    
    Engine->>Output: Write function_calling_results.json

```

## Design Decisions

1. **Pydantic for Validation:** I opted for Pydantic to strictly validate the input JSON schemas (`function_calling_tests.json` and `functions_definition.json`). This ensures that the engine only operates on properly formatted definitions, failing fast if the inputs are malformed.
2. **Astral's `uv` for Dependency Management:** Replaced standard `pip` with `uv` to drastically reduce environment resolution and installation times. The provided `uv.lock` ensures deterministic builds across all environments.
3. **Docker Multi-stage Architecture:** The environment is built on `python:3.12-slim`. To ensure security and prevent file permission issues, the container creates and executes under a non-root user (`appuser`). The HuggingFace cache is mounted as an external volume to avoid re-downloading the model on every run.
4. **Makefile Abstraction:** The complexity of Docker commands, volume mounting, and linting is completely hidden behind a robust `Makefile`, ensuring a smooth developer experience.

## Infrastructure and Volumes

```mermaid
graph TD
    subgraph Host System
        HostDIR[Project Directory]
        HostCache[~/.cache/huggingface]
        Makefile
    end

    subgraph Docker Container: call-me-maybe-dev
        Python[Python 3.12 Slim]
        UV[uv 0.5.11 Package Manager]
        AppUser[appuser: UID 1000]
        AppDIR[/app]
        ContainerCache[/root/.cache/huggingface]
    end

    HostDIR <==>|Mounted Volume -v| AppDIR
    HostCache <==>|Mounted Volume -v| ContainerCache
    Makefile -->|make run| UV
    UV -->|uv run| Python

```

## Performance Analysis

* **Accuracy:** By using constrained decoding, the system achieves near 100% accuracy in syntax generation. As long as the model correctly identifies the semantic intent of the prompt, the resulting JSON will always be structurally flawless.
* **Speed:** Processing speed is highly optimized. While constrained decoding adds a small computational overhead per token (due to vocabulary masking), the use of a small 0.6B parameter model keeps the overall execution well under the 5-minute threshold for the test batch.
* **Reliability:** Standard prompting on small models yields roughly a 30% success rate for valid JSON. This implementation forces 100% JSON validity and schema adherence, making the output entirely deterministic at the structural level.

## Challenges Faced

1. **Tokenization Quirks:** Understanding that LLM tokenizers often prepend spaces (e.g., `Ġ` or raw spaces) to words made filtering valid tokens incredibly difficult. A naive string-matching approach failed; I had to implement an advanced state tracker to handle subword tokens properly.
2. **Logit Manipulation:** Mapping the model's token IDs back to strings in real-time without severe performance degradation required careful caching of the vocabulary file.
3. **Handling Escaped Characters:** Ensuring that the constrained engine allowed for valid JSON string escaping (like quotes inside strings) without breaking the JSON parser was a complex edge case that required strict regex rules during the masking phase.

## Testing Strategy

1. **Static Analysis:** The project relies heavily on `mypy` (with strict flags like `--disallow-untyped-defs`) and `flake8`. This catches type mismatches and syntax errors before runtime.
2. **Unit Testing Edge Cases:** Tested against missing keys, completely malformed JSON files, missing files, and prompts that intentionally try to confuse the LLM (e.g., asking for a calculation when the required function expects string manipulation).
3. **Exception Handling:** Extensive `try-except` blocks are utilized around file I/O and Pydantic validation to ensure the program never crashes unexpectedly, printing human-readable error messages instead of stack traces.

## Instructions

### Prerequisites

* Python 3.10+ (if running locally).
* `make`, `docker`, and `docker-compose` (for containerized execution).

### Installation and Environment Setup

You can run this project using the provided `Makefile` which handles the Docker environment and `uv` package manager automatically.

```bash
# Build the Docker image and create necessary cache directories
make build

# Run the linting checks (flake8 & strict mypy)
make lint
make lint-strict

```

### Execution

To execute the engine, process the inputs, and generate the structured JSON output:

```bash
# Run the project inside the Docker container
make run

```

If you wish to run the project locally without Docker, ensure `uv` is installed and run:

```bash
uv sync
uv run python -m src

```

## Example Usage

The program accepts arguments to override the default input and output paths.

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json

```

**Input Prompt Example:**

```json
{
  "prompt": "What is the sum of 2 and 3?"
}

```

**Generated Output (`function_calls.json`):**

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]

```

### Debugging and Cleanup

```bash
# Run the built-in python debugger (pdb)
make debug

# Clean all caches, compiled files, venvs, and Docker containers/images
make clean

```

## Resources

* **JSON Standard:** [RFC 8259 - The JavaScript Object Notation (JSON) Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259)
* **Constrained Decoding Theory:** [Understanding Constrained Decoding (HuggingFace)](https://huggingface.co/blog/constrained-beam-search)
* **Pydantic Documentation:** [Pydantic V2 Models](https://www.google.com/search?q=https://docs.pydantic.dev/latest/)
* **AI Usage Acknowledgment:** Artificial Intelligence was utilized primarily as a brainstorming tool to conceptualize the regular expressions needed for the token masking logic, and to generate boilerplate structures for the Pydantic schemas. All AI suggestions were rigorously peer-reviewed, heavily modified, and thoroughly tested against the codebase to ensure complete comprehension and accountability.

```

```
