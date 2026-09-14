"""pydantic models for input validation and output formatting.
outputs validated pydantic models.
"""
from pydantic import BaseModel
from typing import Any, Dict


class PromptInput(BaseModel):
    """validates a single test prompt.
    outputs a validated prompt object.
    """

    prompt: str


class ParameterDef(BaseModel):
    """validates a single parameter definition.
    outputs a validated parameter object.
    """

    type: str


class FunctionDef(BaseModel):
    """validates a function definition.
    outputs a validated function object.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: ParameterDef


class FunctionCall(BaseModel):
    """validates the output format for a single function call.
    outputs a validated function call object.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
