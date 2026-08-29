import json
import re
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from openai import APIError


_EMPTY_THINK = re.compile(r"^\s*<think>\s*</think>\s*", re.DOTALL)
T = TypeVar("T")


class StructuredOutputError(RuntimeError):
    """Raised after an LLM repeatedly returns unusable structured output."""


def parse_json_object(text: str, required_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Extract the first JSON object containing the required keys."""
    keys = set(required_keys)
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and keys <= payload.keys():
            return payload
    raise ValueError("No matching JSON object found")


def retry_json_object(
    request: Callable[[], Any],
    required_keys: Iterable[str],
    transform: Callable[[dict[str, Any]], T],
    attempts: int = 3,
) -> T:
    """Request, extract, and validate one JSON object with bounded retries."""
    last_error = None
    for _ in range(attempts):
        try:
            response = request()
            content = response.choices[0].message.content or ""
            return transform(parse_json_object(content, required_keys))
        except (
            APIError,
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
            ValueError,
        ) as error:
            last_error = error
    raise StructuredOutputError(
        f"LLM returned unusable JSON after {attempts} attempts"
    ) from last_error


def is_clean_json_object(text: str) -> bool:
    """Accept JSON alone or after Qwen's empty non-thinking marker."""
    text = _EMPTY_THINK.sub("", text, count=1)
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def clean_model_output(text: str) -> str:
    """Remove Qwen's empty thinking marker from a plain-text response."""
    return _EMPTY_THINK.sub("", text, count=1).strip()
