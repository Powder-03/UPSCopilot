"""Extraction of JSON objects from raw LLM responses (single shared implementation)."""
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _first_json_candidate(text: str) -> str | None:
    """Returns the JSON-looking slice of `text`: a fenced block, or everything from the first '{' to last '}'."""
    if not text:
        return None

    fence = _FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()

    return text[start:].strip() if start != -1 else None


def extract_json_dict(text: str) -> dict[str, Any]:
    """Robustly extracts the primary JSON object from LLM response text, discarding commentary.

    Returns an empty dict when no valid JSON object can be recovered.
    """
    candidate = _first_json_candidate(text)
    if candidate is None:
        return {}

    # Attempt 1: standard raw_decode
    try:
        obj, _ = json.JSONDecoder().raw_decode(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Attempt 2: json.loads with strict=False (allows unescaped newlines/tabs in string values)
    try:
        obj = json.loads(candidate, strict=False)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Attempt 3: slice from first { to last }
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end > start:
        slice_str = candidate[start : end + 1]
        try:
            obj = json.loads(slice_str, strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Attempt 4: Clean comments and trailing commas from the slice
        try:
            cleaned = re.sub(r"^\s*//.*$", "", slice_str, flags=re.MULTILINE)
            cleaned = re.sub(r"\s+//.*$", "", cleaned, flags=re.MULTILINE)
            cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
            cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
            obj = json.loads(cleaned, strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            logger.warning(f"Error parsing JSON from sanitized LLM output: {e}")

    # Attempt 5: Handle truncated JSON by closing open strings and braces
    if candidate.startswith("{"):
        repaired = candidate
        if repaired.count('"') % 2 != 0:
            repaired += '"'
        open_braces = repaired.count("{") - repaired.count("}")
        if open_braces > 0:
            repaired += "}" * open_braces
        try:
            obj = json.loads(repaired, strict=False)
            if isinstance(obj, dict):
                logger.info("Successfully recovered truncated JSON via brace closure")
                return obj
        except Exception:
            pass

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
