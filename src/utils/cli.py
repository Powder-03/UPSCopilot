"""Console and logging setup shared by every operational script.

Replaces the win32 encoding shim and logging boilerplate that was copy-pasted
across the evaluate_* and ingest_* scripts.
"""
import logging
import sys
from collections.abc import Sequence

# Third-party SDKs that are extremely chatty at INFO level during Bedrock calls.
DEFAULT_QUIET_LOGGERS: Sequence[str] = ("langchain_aws", "boto3", "botocore")


def configure_console() -> None:
    """Forces UTF-8 stdio on Windows so Unicode output (quotes, symbols) cannot crash a run."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass


def setup_logging(level: int = logging.INFO, quiet: Sequence[str] = ()) -> None:
    """Configures root logging and silences noisy SDK loggers.

    Args:
        level: Root log level for the script's own modules.
        quiet: Extra logger names to pin at WARNING, beyond the default SDK set.
    """
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    for name in (*DEFAULT_QUIET_LOGGERS, *quiet):
        logging.getLogger(name).setLevel(logging.WARNING)
