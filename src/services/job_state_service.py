"""Pluggable job state persistence service supporting local JSON files and Amazon DynamoDB."""
import decimal
import json
import logging
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def _floats_to_decimals(obj: Any) -> Any:
    """Recursively converts float values to Decimal for DynamoDB serialization."""
    if isinstance(obj, float):
        return decimal.Decimal(str(obj))
    if isinstance(obj, dict):
        return {k: _floats_to_decimals(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_floats_to_decimals(v) for v in obj]
    return obj


def _decimals_to_floats(obj: Any) -> Any:
    """Recursively converts Decimal values back to float/int after DynamoDB read."""
    if isinstance(obj, decimal.Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    if isinstance(obj, dict):
        return {k: _decimals_to_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decimals_to_floats(v) for v in obj]
    return obj


class JobStateService:
    """Manages job state records, abstracting local disk JSON files vs Amazon DynamoDB."""

    def __init__(
        self,
        local_dir: str | Path | None = None,
        dynamodb_table: str | None = None,
        region_name: str | None = None,
    ):
        self.local_dir = Path(local_dir or settings.job_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.table_name = dynamodb_table if dynamodb_table is not None else settings.dynamodb_table
        self.region_name = region_name or settings.aws_region
        self._table: Any = None

    @property
    def is_dynamodb_enabled(self) -> bool:
        """Returns True if Amazon DynamoDB table is configured."""
        return bool(self.table_name and self.table_name.strip())

    @property
    def table(self) -> Any:
        """Lazy instantiation of boto3 DynamoDB Table resource."""
        if self._table is None:
            import boto3

            dynamodb = boto3.resource("dynamodb", region_name=self.region_name)
            self._table = dynamodb.Table(self.table_name)
        return self._table

    def save_job(self, job_id: str, job_data: dict[str, Any]) -> None:
        """Saves job state to DynamoDB or local JSON file."""
        if self.is_dynamodb_enabled:
            try:
                # DynamoDB requires float -> Decimal conversion
                item = _floats_to_decimals(job_data)
                # Ensure primary key is explicitly present
                item["job_id"] = job_id
                self.table.put_item(Item=item)
                logger.debug("Saved job %s to DynamoDB table %s", job_id, self.table_name)
            except Exception as e:
                logger.error("Failed to save job %s to DynamoDB: %s", job_id, e)
                raise

        # Always maintain local disk copy if local directory is writable
        try:
            job_file = self.local_dir / f"{job_id}.json"
            with open(job_file, "w", encoding="utf-8") as f:
                json.dump(job_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("Skipped local disk save for job %s: %s", job_id, e)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Loads job state from DynamoDB or local JSON file."""
        if self.is_dynamodb_enabled:
            try:
                response = self.table.get_item(Key={"job_id": job_id})
                item = response.get("Item")
                if item:
                    return _decimals_to_floats(item)
                return None
            except Exception as e:
                logger.error("Failed to fetch job %s from DynamoDB: %s", job_id, e)
                # Fallback to local disk if DynamoDB read fails
                pass

        # Local filesystem mode
        job_file = self.local_dir / f"{job_id}.json"
        if job_file.exists():
            try:
                with open(job_file, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to read job %s from local disk: %s", job_id, e)
                return None

        return None
