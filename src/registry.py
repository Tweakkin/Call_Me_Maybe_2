"""loads and manages function definitions.
outputs a loaded registry object.
"""
import json
from typing import Any, Dict, List
from pydantic import BaseModel

from src.models import FunctionDef


class FunctionRegistry(BaseModel):
    """stores and provides access to function definitions.
    outputs nothing directly.
    """
    functions: Dict[str, FunctionDef] = {}

    def load(self, path: str) -> None:
        """loads function definitions from a json file.
        outputs nothing.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for fn in data:
            validated = FunctionDef.model_validate(fn)
            self.functions[validated.name] = validated

    def get_functions_name(self) -> List[str]:
        """gets all registered function names.
        outputs a list of strings.
        """
        return list(self.functions.keys())

    def get_description(self, name: str) -> str:
        """gets the description of a function.
        outputs a string description.
        """
        fn = self.functions.get(name)
        if fn is None:
            return ""
        return fn.description

    def get_parameters(self, name: str) -> Dict[str, Any]:
        """gets parameter info for a function.
        outputs a dictionary of parameters.
        """
        fn = self.functions.get(name)
        if fn is None:
            return {}
        return {
            k: v.model_dump() for k, v in fn.parameters.items()
        }

    def get_returns(self, name: str) -> Dict[str, Any]:
        """gets return info for a function.
        outputs a dictionary of return types.
        """
        fn = self.functions.get(name)
        if fn is None:
            return {}
        return fn.returns.model_dump()
