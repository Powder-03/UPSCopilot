"""Pluggable file storage service supporting local filesystem and Amazon S3."""
import logging
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """Manages file storage, abstracting local disk vs Amazon S3 transparently."""

    def __init__(
        self,
        local_dir: str | Path | None = None,
        s3_bucket: str | None = None,
        region_name: str | None = None,
    ):
        self.local_dir = Path(local_dir or settings.upload_dir)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.s3_bucket = s3_bucket if s3_bucket is not None else settings.s3_bucket
        self.region_name = region_name or settings.aws_region
        self._s3_client: Any = None

    @property
    def is_s3_enabled(self) -> bool:
        """Returns True if Amazon S3 bucket is configured."""
        return bool(self.s3_bucket and self.s3_bucket.strip())

    @property
    def s3_client(self) -> Any:
        """Lazy instantiation of boto3 S3 client."""
        if self._s3_client is None:
            import boto3

            self._s3_client = boto3.client("s3", region_name=self.region_name)
        return self._s3_client

    def generate_presigned_upload_url(
        self, filename: str, job_id: str, expires_in: int = 3600
    ) -> dict[str, Any]:
        """
        Generates a presigned S3 PUT URL allowing the client browser to upload
        large files directly to S3, bypassing API Gateway's 10 MB payload ceiling.
        """
        clean_filename = Path(filename).name
        if self.is_s3_enabled:
            s3_key = f"uploads/{job_id}_{clean_filename}"
            url = self.s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.s3_bucket,
                    "Key": s3_key,
                    "ContentType": "application/pdf",
                },
                ExpiresIn=expires_in,
            )
            return {
                "job_id": job_id,
                "upload_url": url,
                "storage_ref": f"s3://{self.s3_bucket}/{s3_key}",
                "s3_enabled": True,
            }

        return {
            "job_id": job_id,
            "upload_url": "",
            "storage_ref": "",
            "s3_enabled": False,
        }

    def save_file(self, file_bytes: bytes, filename: str, job_id: str) -> str:
        """
        Saves uploaded file bytes.
        - If S3 is configured, uploads to S3 bucket under 'uploads/{job_id}_{filename}'.
        - Otherwise, saves to local upload_dir.
        Returns the storage path or S3 key string.
        """
        clean_filename = Path(filename).name
        local_path = self.local_dir / f"{job_id}_{clean_filename}"

        if self.is_s3_enabled:
            s3_key = f"uploads/{job_id}_{clean_filename}"
            logger.info("Uploading %d bytes to S3: s3://%s/%s", len(file_bytes), self.s3_bucket, s3_key)
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=file_bytes,
                ContentType="application/pdf",
            )
            # Also write locally if directory is writable (e.g. for fast local access)
            try:
                local_path.write_bytes(file_bytes)
            except Exception as e:
                logger.debug("Skipped local cache write: %s", e)

            return f"s3://{self.s3_bucket}/{s3_key}"

        # Local filesystem mode
        local_path.write_bytes(file_bytes)
        logger.info("Saved %d bytes to local disk: %s", len(file_bytes), local_path)
        return str(local_path)

    def get_local_path(self, storage_ref: str, temp_dir: str | Path | None = None) -> Path:
        """
        Resolves a storage reference (local path or s3:// URI) to a local Path on disk.
        If reference is in S3, downloads to temp_dir (defaults to /tmp or local_dir).
        """
        if storage_ref.startswith("s3://"):
            parts = storage_ref[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            target_dir = Path(temp_dir or "/tmp" if Path("/tmp").exists() else self.local_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            local_dest = target_dir / Path(key).name

            if not local_dest.exists() or local_dest.stat().st_size == 0:
                logger.info("Downloading s3://%s/%s to local %s", bucket, key, local_dest)
                self.s3_client.download_file(bucket, key, str(local_dest))

            return local_dest

        # Otherwise treated as local path
        return Path(storage_ref)
