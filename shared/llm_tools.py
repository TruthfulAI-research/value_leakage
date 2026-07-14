"""Shared local LLM tools used by experiments and the playground.

This file is the source of truth for the model-visible tool names,
descriptions, schemas, and Python handlers.
"""

from __future__ import annotations

import json
import random
import string
import time
from typing import Any


_RNG = random.SystemRandom()


def _coin_flip(_args: dict[str, Any]) -> dict[str, str]:
    """Return one fair coin flip."""
    return {"result": _RNG.choice(["heads", "tails"])}


def _random_letter(_args: dict[str, Any]) -> dict[str, str]:
    """Return one random uppercase ASCII letter."""
    return {"letter": _RNG.choice(string.ascii_uppercase)}


def _random_token(_args: dict[str, Any]) -> dict[str, str]:
    """Return one short random alphanumeric token."""
    alphabet = string.ascii_lowercase + string.digits
    return {"token": "".join(_RNG.choice(alphabet) for _ in range(8))}


def _time_seconds(_args: dict[str, Any]) -> dict[str, float]:
    """Return a random Unix timestamp from the previous two seconds."""
    now = time.time()
    return {"unix_seconds": round(_RNG.uniform(now - 2.0, now), 6)}


_EMPTY_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "required": [],
    "additionalProperties": False,
}


LLM_TOOLS: dict[str, dict[str, Any]] = {
    "coin_flip": {
        "name": "coin_flip",
        "label": "coin_flip",
        "description": (
            "Flip a fair coin once."
        ),
        "parameters": _EMPTY_OBJECT_SCHEMA,
        "strict": True,
        "handler": _coin_flip,
    },
    "coin_flip_blank": {
        "name": "coin_flip_blank",
        "label": "coin_flip",
        "description": "",
        "schema_description": None,
        "parameters": _EMPTY_OBJECT_SCHEMA,
        "strict": True,
        "handler": _coin_flip,
    },
    "random_letter": {
        "name": "random_letter",
        "label": "random_letter",
        "description": (
            "Generate one random uppercase letter. Use when the user asks for "
            "a random letter or when a random choice can be mapped from a "
            "letter."
        ),
        "parameters": _EMPTY_OBJECT_SCHEMA,
        "strict": True,
        "handler": _random_letter,
    },
    "random_token": {
        "name": "random_token",
        "label": "random_token",
        "description": (
            "Generate a short random alphanumeric token: a string whose "
            "characters are randomly chosen from lowercase letters and digits."
        ),
        "parameters": _EMPTY_OBJECT_SCHEMA,
        "strict": True,
        "handler": _random_token,
    },
    "time_seconds": {
        "name": "time_seconds",
        "label": "time_seconds",
        "description": (
            "Return the Unix timestamp in seconds."
        ),
        "parameters": _EMPTY_OBJECT_SCHEMA,
        "strict": True,
        "handler": _time_seconds,
    },
}


# The local handlers are pure Python. This list tracks the model-provider
# dispatchers that know how to run a tool loop and feed outputs back.
LLM_TOOL_BACKENDS = ["openai", "claude"]


def list_llm_tools() -> list[dict[str, Any]]:
    """Return frontend/API metadata for all locally implemented tools."""
    return [
        {
            "name": tool["name"],
            "label": tool["label"],
            "backends": list(LLM_TOOL_BACKENDS),
            "backend": LLM_TOOL_BACKENDS[0],
            "description": tool.get("description") or "",
        }
        for tool in LLM_TOOLS.values()
    ]


def openai_tool_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    """Return OpenAI Responses API function-tool schemas."""
    schemas = []
    for name in tool_names:
        tool = LLM_TOOLS.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        schema = {
            "type": "function",
            "name": tool["name"],
            "parameters": tool["parameters"],
            "strict": tool["strict"],
        }
        description = tool.get("schema_description", tool.get("description"))
        if description is not None:
            schema["description"] = description
        schemas.append(schema)
    return schemas


def claude_tool_schemas(tool_names: list[str]) -> list[dict[str, Any]]:
    """Return Anthropic tool schemas for the same local tools."""
    schemas = []
    for name in tool_names:
        tool = LLM_TOOLS.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        schema = {
            "name": tool["name"],
            "input_schema": tool["parameters"],
        }
        description = tool.get("schema_description", tool.get("description"))
        if description is not None:
            schema["description"] = description
        schemas.append(schema)
    return schemas


def run_llm_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Execute one local tool by name."""
    tool = LLM_TOOLS.get(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    return tool["handler"](args)


def format_tool_event(name: str, args: Any, output: dict[str, Any], error: str | None) -> str:
    """Human-readable trace block for reasoning displays and saved rows."""
    if isinstance(args, dict):
        args_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
    else:
        args_text = str(args)
    output_text = json.dumps(output, ensure_ascii=False, sort_keys=True)
    lines = [
        f"[tool call] {name}",
        f"arguments: {args_text}",
        f"output: {output_text}",
    ]
    if error:
        lines.append(f"error: {error}")
    return "\n".join(lines)
