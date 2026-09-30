"""Extraction of JSON objects from raw LLM responses (single shared implementation)."""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _first_json_candidate(text: str) -> str | None:
    """Returns the JSON-looking slice of `text`: a fenced block, or everything from the first '{'."""
    if not text:
        return None

    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    start = text.find("{")
    return text[start:] if start != -1 else None


def extract_json_dict(text: str) -> dict[str, Any]:
    """Robustly extracts the primary JSON object from LLM response text, discarding commentary.

    Returns an empty dict when no valid JSON object can be recovered.
    """
    candidate = _first_json_candidate(text)
    if candidate is None:
        return {}

    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        logger.warning(f"Error parsing JSON from LLM output: {e}")

    return {}


def clean_json_text(text: str) -> str:
    """Returns a compact JSON string for downstream consumers (e.g. DeepEval judges).

    Falls back to the raw candidate slice, or "{}" for empty input, so callers always
    receive a string rather than a parse failure.
    """
    candidate = _first_json_candidate(text)
    if candidate is None:
        return text or "{}"

    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate)
        return json.dumps(obj)
    except Exception:
        return candidate
