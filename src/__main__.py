"""main entry point for the function calling system.
outputs structured json function calls.
"""
import argparse
import os
import sys
import numpy as np
import json
from typing import Any, Dict, List, cast
from pydantic import BaseModel


from llm_sdk import Small_LLM_Model

from src.models import PromptInput, FunctionCall
from src.registry import FunctionRegistry


def build_prompt(
    registry: FunctionRegistry, user_prompt: str
) -> str:
    """builds the prompt with available functions.
    outputs the formatted prompt string.
    """
    functions_list = []
    for fn_name in registry.get_functions_name():
        desc = registry.get_description(fn_name)
        params = registry.get_parameters(fn_name)
        returns = registry.get_returns(fn_name).get("type", "")

        params_str = ", ".join(
            f"{k}: {v.get('type', 'string')}" for k, v in params.items()
        )
        line = (
            f"function_name: {fn_name}({params_str}) "
            f":returns({returns}) :description ({desc})"
        )
        functions_list.append(line)

    fallback_line = "function_name: fn_uknown"
    functions_list.append(fallback_line)
    functions_str = "\n".join(functions_list)

    return (
        f"Available functions:\n{functions_str}\n\n"
        f"User request: {user_prompt}\n\n"
        "- For regex parameters, generate a valid regex "
        "- Return fn_uknown only if none of the listed "
        "functions can handle the user prompt.\n\n"
        "pattern (e.g use [...] not ...)\n\n"
    )


class MainEngine(BaseModel):
    """generates function calls using constrained decoding.
    outputs structured json dictionaries.
    """
    model_config = {"arbitrary_types_allowed": True}
    ai: Any
    registry: FunctionRegistry

    def _encode_into_ids(self, text: str) -> List[int]:
        """encodes text into tokens.
        outputs a list of token ids.
        """
        res = self.ai.encode(text).squeeze().tolist()
        return res if isinstance(res, list) else [res]

    def _get_logits(self, input_ids: List[int]) -> List[float]:
        """gets probabilities for the next token.
        outputs a list of floats.
        """
        return cast(List[float], self.ai.get_logits_from_input_ids(input_ids))

    def _pick_correct_function(self, input_ids: List[int]) -> str:
        """forces the ai to pick a valid function name.
        outputs the chosen function name string.
        """
        all_funcs_tokens = [
            self._encode_into_ids(name)
            for name in self.registry.get_functions_name() + ["fn_uknown"]
        ]
        candidates = list(all_funcs_tokens)
        selected_tokens = []
        token_position = 0

        while candidates:
            allowed_tokens = set()
            for func_tokens in candidates:
                if token_position < len(func_tokens):
                    allowed_tokens.add(func_tokens[token_position])

            if not allowed_tokens:
                break
            if len(allowed_tokens) == 1:
                token = next(iter(allowed_tokens))
            else:
                logits = np.array(self._get_logits(input_ids))
                for token_id in range(len(logits)):
                    if token_id not in allowed_tokens:
                        logits[token_id] = -float('inf')
                token = int(np.argmax(logits))
            selected_tokens.append(token)
            input_ids.append(token)
            candidates = [
                func_tokens for func_tokens in candidates
                if token_position < len(func_tokens)
                and func_tokens[token_position] == token
            ]
            token_position += 1
            for func_tokens in all_funcs_tokens:
                if selected_tokens == func_tokens:
                    result = cast(str, self.ai.decode(selected_tokens)).strip()
                    input_ids.extend(self._encode_into_ids('"'))
                    return result
        return "fn_uknown"

    def _generate_string_until_quote(self, input_ids: List[int]) -> str:
        """generates string tokens until a closing quote.
        outputs the decoded string.
        """
        generated_string_tokens = []
        input_ids.extend(self._encode_into_ids(' "'))
        previous_token_text = None

        while True:
            logits = self._get_logits(input_ids)
            token = int(np.argmax(logits))
            token_text = cast(str, self.ai.decode([token]))

            if '"' in token_text and previous_token_text != "\\":
                new_text = token_text.split('"')[0]
                if new_text:
                    tokens = self._encode_into_ids(new_text)
                    input_ids.extend(tokens)
                    generated_string_tokens.extend(tokens)
                break

            input_ids.append(token)
            generated_string_tokens.append(token)
            previous_token_text = token_text

        decoded_string_text = cast(
            str, self.ai.decode(generated_string_tokens)
        ).strip()
        decoded_string_text = decoded_string_text.replace("\\\\", "\\")
        input_ids.extend(self._encode_into_ids('"'))
        return decoded_string_text

    def _generate_number_until_comma(
        self, input_ids: List[int], is_integer: bool
    ) -> float | int:
        """generates numeric tokens until a comma or brace.
        outputs the parsed float or integer.
        """
        generated_number_tokens = []
        while True:
            logits = self._get_logits(input_ids)
            token = int(np.argmax(logits))
            token_text = cast(str, self.ai.decode([token]))

            if any(c in token_text for c in {",", "}"}):
                digit_part = token_text.split(",")[0].split("}")[0].strip()
                if digit_part:
                    input_ids.extend(self._encode_into_ids(digit_part))
                break

            input_ids.append(token)
            generated_number_tokens.append(token)

        decoded_number_text = cast(str, self.ai.decode(
            generated_number_tokens
        )).strip().strip('"')
        if is_integer:
            return int(decoded_number_text)
        return float(decoded_number_text)

    def _force_true_false_choice(self, input_ids: List[int]) -> bool:
        """forces the model to predict true or false.
        outputs the boolean result.
        """
        true_tokens = self._encode_into_ids("true")
        false_tokens = self._encode_into_ids("false")

        logits = np.array(self._get_logits(input_ids))
        allowed = {true_tokens[0], false_tokens[0]}

        for token_id in range(len(logits)):
            if token_id not in allowed:
                logits[token_id] = -float('inf')

        first_token = int(np.argmax(logits))
        if first_token == true_tokens[0]:
            input_ids.extend(true_tokens)
            return True

        input_ids.extend(false_tokens)
        return False

    def generate(self, question: str) -> Dict[str, Any]:
        """generates a complete function call.
        outputs the final json parameters.
        """
        prompt_text = build_prompt(self.registry, question)
        # Force JSON response
        prompt_text += 'JSON response:\n{"function_name": "'

        input_ids = self._encode_into_ids(prompt_text)

        function_name = self._pick_correct_function(input_ids)

        if function_name == "fn_uknown":
            return {
                "prompt": question,
                "name": "fn_not_found",
                "parameters": {}
            }

        # {"function_name": "fn_add_numbers", "parameters": {
        input_ids.extend(self._encode_into_ids('", "parameters": {'))
        curr_func_params = self.registry.get_parameters(function_name)
        result_params: Dict[str, Any] = {}

        for i, (param_name, param_info) in enumerate(curr_func_params.items()):
            if i > 0:
                input_ids.extend(self._encode_into_ids(', '))

            input_ids.extend(self._encode_into_ids(f'"{param_name}":'))
            ptype = param_info.get("type", "string")

            if ptype == "string":
                result_params[param_name] = self._generate_string_until_quote(
                    input_ids
                )
            elif ptype in ["number", "integer"]:
                result_params[param_name] = self._generate_number_until_comma(
                    input_ids, ptype == "integer"
                )
            elif ptype == "boolean":
                result_params[param_name] = self._force_true_false_choice(
                    input_ids
                )

        # Force the final closing brace into input_ids
        input_ids.extend(self._encode_into_ids('}}'))

        return {
            "prompt": question,
            "name": function_name,
            "parameters": result_params
        }


def generate_function_call(
    ai: Any,
    registry: FunctionRegistry,
    question: str,
) -> Dict[str, Any]:
    """generates a single function call.
    outputs a dictionary with prompt, name, and parameters.
    """
    engine = MainEngine(ai=ai, registry=registry)
    result = engine.generate(question)
    print(json.dumps(result))
    return result


def main() -> None:
    """loads files, processes prompts, and writes results.
    outputs nothing directly.
    """
    # handle command line arguments and --help
    parser = argparse.ArgumentParser(
        description="Function calling with constrained decoding."
    )
    # then Handle "--functions_definitions"
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON.",
    )
    # then Handle "--input"
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON.",
    )
    # then we handle "--output"
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path for the output JSON file.",
    )
    # return an object with parsed properties
    args = parser.parse_args()

    try:
        # Load functions and validate their format using Pydantic
        registry = FunctionRegistry()
        registry.load(args.functions_definition)

        with open(args.input, "r", encoding="utf-8") as f:
            # [ {"prompt": "What is 2+2?"}, {"prompt": "Say hello"} ]
            raw_tests = json.load(f)
        prompts = [PromptInput.model_validate(t) for t in raw_tests]
    except Exception as e:
        print(f"Error: Failed to load input files.\n{e}")
        sys.exit(0)

    print("Initializing LLM...")
    ai = Small_LLM_Model()

    results: List[Dict[str, Any]] = []

    # Main Engine Loop, Goes through prompts One by One
    for i, prompt in enumerate(prompts):
        print(f"\n--- Test {i + 1}/{len(prompts)}: {prompt.prompt} ---")

        try:
            result = generate_function_call(
                ai, registry, prompt.prompt
            )
            FunctionCall.model_validate(result)
            results.append(result)
            print("SUCCESS")
        except Exception as e:
            print(f"Error processing prompt: {e}")
            results.append({
                "prompt": prompt.prompt,
                "name": "fn_not_found",
                "parameters": {}
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nWrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Exiting cleanly.")
        sys.exit(0)
