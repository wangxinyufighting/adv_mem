import json

from typing import Any, Iterable


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


def is_exact_json_object(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False
