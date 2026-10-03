"""LangSmith tracing utilities, lifecycle flushes for AWS Lambda, and payload redactions."""
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from src.config import settings

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_client = None


def get_langsmith_client() -> Any | None:
    """Returns singleton LangSmith Client if configured and available."""
    global _client
    if _client is not None:
        return _client

    if not settings.langsmith_api_key or not settings.langsmith_tracing:
        return None

    try:
        from langsmith import Client

        _client = Client(
            api_key=settings.langsmith_api_key,
            api_url=settings.langsmith_endpoint,
        )
        return _client
    except Exception as e:
        logger.debug("Could not initialize LangSmith Client: %s", e)
        return None


def flush_traces(timeout: float = 5.0) -> None:
    """
    Forces all buffered background trace batches to post to LangSmith immediately.

    Critical for AWS Lambda environments (e.g. vision_worker, eval_worker) where
    the runtime CPU freezes as soon as the handler returns, which would otherwise
    drop or stall pending background HTTP trace emissions.
    """
    if not settings.langsmith_tracing or not settings.langsmith_api_key:
        return

    try:
        # 1. Flush the exact cached client used by @traceable and RunTree
        try:
            from langsmith.run_trees import get_cached_client

            cached_c = get_cached_client()
            if cached_c and hasattr(cached_c, "flush"):
                cached_c.flush()
        except Exception as e:
            logger.debug("Cached client flush notice: %s", e)

        # 2. Flush custom singleton client if active
        client = get_langsmith_client()
        if client and hasattr(client, "flush"):
            client.flush()
        logger.debug("Successfully flushed LangSmith traces (timeout=%ss)", timeout)
    except Exception as e:
        logger.debug("LangSmith trace flush completed with notice: %s", e)


def mask_vision_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Redacts large binary page image payloads from LangSmith trace inputs.

    Prevents megabytes of raw image bytes or base64 strings from inflating
    trace sizes and causing network lag or HTTP payload size limits.
    """
    if not isinstance(inputs, dict):
        return {"input": str(inputs)[:200]}

    cleaned: dict[str, Any] = {}
    for key, val in inputs.items():
        if key == "image_bytes_list" and isinstance(val, (list, tuple)):
            cleaned["page_count"] = len(val)
            cleaned["total_image_bytes"] = sum(len(b) for b in val if isinstance(b, bytes))
        elif key == "self":
            # Omit self instance references from traces
            continue
        elif isinstance(val, bytes):
            cleaned[key] = f"<binary data: {len(val)} bytes>"
        else:
            cleaned[key] = val

    return cleaned


# Re-export traceable decorator from langsmith with safe fallback
try:
    from langsmith import traceable
except ImportError:
    # No-op fallback decorator if langsmith is ever absent
    def traceable(
        name: str | None = None,
        run_type: str | None = None,
        process_inputs: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator


__all__ = [
    "flush_traces",
    "get_langsmith_client",
    "mask_vision_inputs",
    "traceable",
]
