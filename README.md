*This project has been created as part of the 42 curriculum by your_login.*

# Call Me Maybe - Constrained Decoding for LLM Function Calling

## Description
This project implements a system for Large Language Models (LLMs) to natively output valid JSON function calls without relying on regex parsing or JSON repairing. By intercepting the LLM's next-token generation process, the system dynamically masks invalid tokens out of the model's vocabulary (setting their logits to `-infinity`), mathematically forcing the LLM to output structurally perfect JSON keys, values, and syntax.

## Instructions
1. Install dependencies via `make install`.
2. Ensure you have the `llm_sdk` directory correctly placed.
3. Place your test prompts in `data/input/function_calling_tests.json`.
4. Place your function definitions in `data/input/functions_definition.json`.
5. Run the engine with `make run` or `uv run python -m src`.
6. Results will be saved to `data/output/function_calling_results.json`.

## Algorithm Explanation
The constrained decoding algorithm works by intercepting the LLM's raw output logits before a token is finalized:
1. **State Tracking**: The `MainEngine` tracks exactly where we are in the JSON structure (e.g., generating the function name vs. generating a specific parameter value).
2. **Logit Masking**: We maintain sets of "allowed tokens" for the current state. For example, when generating a boolean parameter, only the tokens for "true" and "false" are allowed. The logits for all 150,000+ other tokens in the model's vocabulary are overridden to `-inf`.
3. **Forced JSON Framing**: The structural braces (`{`, `}`, `", "parameters": {`, etc.) are entirely forced. The LLM does not predict them; they are injected directly into the context window, so the LLM is only tasked with filling in the blanks.

## Design Decisions
- **Token-by-Token Trie Traversal for Function Names**: To guarantee the LLM selects a valid function name from the `FunctionRegistry`, we treat the valid function names as a collection of token sequences. At each step, we look at which functions are still viable candidates based on the tokens generated so far, and only allow the LLM to pick tokens that continue those viable paths.
- **Pydantic Validation**: `pydantic.BaseModel` is used across `models.py` and the main engine for strict type validation of inputs and function definitions, enforcing the data constraints specified in the subject.
- **Greedy Decoding**: Since we heavily constrain the output space, we rely entirely on greedy decoding (`np.argmax`) rather than sampling, ensuring maximum determinism and reliability.

## Performance Analysis
The bottleneck in generation is typically the forward pass of the LLM. Because we override logits purely via array indexing and masking in Python using `numpy`, the overhead of the constraint logic itself is on the order of microseconds per token—negligible compared to the milliseconds required by the LLM inference step. The time complexity of evaluating allowed tokens scales with the vocabulary size (which we optimize by using set membership testing).

## Challenges Faced
- **Tokenization Anomalies**: Different text encodings (like `true` vs ` true`) map to completely different token IDs. We had to ensure our forced dictionary options perfectly aligned with how the LLM's tokenizer splits prefixes and whitespaces.
- **Typo preservation for Model Stability**: The Qwen 0.6B model proved incredibly sensitive to the exact wording and whitespace of the system prompt. Even correcting simple typos in the prompt caused the model to drastically shift its internal representations and hallucinate parameters.

## Testing Strategy
- The system was tested iteratively against both the public and private `Moulinette` testing suites, achieving a 100% success rate on both.
- Unit tests cover various parameter types including `string`, `integer`, `number`, and `boolean`, as well as fallback generation when a prompt matches none of the available functions.

## Example Usage
```bash
$ make install
$ make run
--- Test 1: What is the product of 3 and 5? ---
{"function_name": "fn_multiply_numbers", "parameters": {"a": 3, "b": 5}}
SUCCESS
```

## Resources
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Understanding Constrained Decoding](https://arxiv.org/abs/2307.09702)
