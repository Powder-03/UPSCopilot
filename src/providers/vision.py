"""Multimodal Vision Client for extracting handwritten text, diagrams, and QCAB layouts supporting Bedrock and Vertex AI."""
import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from src.config import settings
from src.models.exceptions import ModelInvocationError
from src.parsing.prompts import (
    VISION_QCAB_PARSER_SYSTEM_PROMPT,
    VISION_SINGLE_PAGE_OCR_SYSTEM_PROMPT,
)
from src.utils.json import extract_json_dict
from src.utils.tracing import mask_vision_inputs, traceable

logger = logging.getLogger(__name__)


class BaseVisionClient(ABC):
    """Abstract base class for multimodal handwriting and diagram extraction clients."""

    @abstractmethod
    def extract_answer_from_page_images(
        self,
        image_bytes_list: list[bytes],
        image_format: str = "jpeg",
        fallback_question: str | None = None,
        expected_q_num: int | None = None,
    ) -> dict[str, Any]:
        """Sends page images to the multimodal vision model and extracts structured question/answer JSON."""
        pass

    @abstractmethod
    def parse_single_page(
        self,
        image_bytes: bytes,
        page_num: int,
        image_format: str = "jpeg",
    ) -> dict[str, Any]:
        """Parses a single page: detects question header, transcribes content, classifies front/back matter."""
        pass


class BedrockVisionClient(BaseVisionClient):
    """Invokes AWS Bedrock Multimodal Vision models via Converse API."""

    def __init__(
        self,
        model_id: str | None = None,
        region_name: str | None = None,
        client: Any = None,
    ):
        self.region_name = region_name or settings.aws_region
        self.model_id = model_id or settings.bedrock_vision_model_id
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    @traceable(name="Bedrock_Multimodal_Vision_OCR", run_type="llm", process_inputs=mask_vision_inputs)
    def extract_answer_from_page_images(
        self,
        image_bytes_list: list[bytes],
        image_format: str = "jpeg",
        fallback_question: str | None = None,
        expected_q_num: int | None = None,
    ) -> dict[str, Any]:
        """Sends page images to Bedrock Vision (Kimi 2.5) and extracts handwritten answer JSON."""
        if not image_bytes_list:
            raise ValueError("image_bytes_list cannot be empty.")

        content_blocks: list[dict[str, Any]] = []
        for img_bytes in image_bytes_list:
            content_blocks.append({
                "image": {
                    "format": image_format.lower().replace("jpg", "jpeg"),
                    "source": {"bytes": img_bytes},
                }
            })

        user_instruction = (
            f"Please transcribe this answer attempt. "
            f"Total scanned pages provided: {len(image_bytes_list)}."
        )
        if expected_q_num:
            user_instruction += f" Expected Question Number: {expected_q_num}."
        if fallback_question:
            user_instruction += f" Reference Question Text (for verification): '{fallback_question}'."

        content_blocks.append({"text": user_instruction})

        messages = [
            {
                "role": "user",
                "content": content_blocks,
            }
        ]

        logger.info(
            f"Calling Bedrock Vision ({self.model_id}) with {len(image_bytes_list)} page images..."
        )

        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": VISION_QCAB_PARSER_SYSTEM_PROMPT}],
                messages=messages,
                inferenceConfig={
                    "temperature": 0.1,
                    "maxTokens": 4096,
                },
            )
            raw_text = response["output"]["message"]["content"][0]["text"]
            parsed_json = extract_json_dict(raw_text)
            return parsed_json
        except Exception as e:
            logger.error(f"Error calling Bedrock Vision: {e}")
            raise ModelInvocationError(f"Bedrock Vision OCR failed: {e}") from e

    @traceable(name="Bedrock_Single_Page_OCR", run_type="llm", process_inputs=mask_vision_inputs)
    def parse_single_page(
        self,
        image_bytes: bytes,
        page_num: int,
        image_format: str = "jpeg",
    ) -> dict[str, Any]:
        """Parses a single page to dynamically detect question boundaries and transcribe handwritten content."""
        content_blocks = [
            {
                "image": {
                    "format": image_format.lower().replace("jpg", "jpeg"),
                    "source": {"bytes": image_bytes},
                }
            },
            {"text": f"Please transcribe Page {page_num} and detect if it starts a new question."},
        ]
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": VISION_SINGLE_PAGE_OCR_SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": content_blocks}],
                inferenceConfig={"temperature": 0.1, "maxTokens": 4096},
            )
            raw_text = response["output"]["message"]["content"][0]["text"]
            parsed_json = extract_json_dict(raw_text)
            parsed_json["page_num"] = page_num
            return parsed_json
        except Exception as e:
            logger.error("Error in Bedrock single page OCR on Page %d: %s", page_num, e)
            raise ModelInvocationError(f"Bedrock single page OCR failed on Page {page_num}: {e}") from e


class VertexVisionClient(BaseVisionClient):
    """Invokes Google Cloud Vertex AI Multimodal Vision (Gemini 2.5 Flash) via official google-genai SDK."""

    def __init__(
        self,
        model_id: str | None = None,
        project_id: str | None = None,
        location: str | None = None,
        api_key: str | None = None,
        client: Any = None,
    ):
        self.model_id = model_id or settings.vertex_vision_model_id
        self.project_id = project_id or settings.gcp_project_id
        self.location = location or settings.gcp_location
        self.api_key = api_key or settings.gemini_api_key or ""
        self._explicit_client = client
        self._thread_local = threading.local()

    @property
    def client(self):
        if self._explicit_client is not None:
            return self._explicit_client
        if not hasattr(self._thread_local, "client") or self._thread_local.client is None:
            from google import genai
            self._thread_local.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                api_key=self.api_key,
            )
        return self._thread_local.client

    def _reset_thread_client(self) -> None:
        """Resets the thread-local client in case an HTTP transport connection closed."""
        if hasattr(self._thread_local, "client"):
            self._thread_local.client = None

    @traceable(name="Vertex_Multimodal_Vision_OCR", run_type="llm", process_inputs=mask_vision_inputs)
    def extract_answer_from_page_images(
        self,
        image_bytes_list: list[bytes],
        image_format: str = "jpeg",
        fallback_question: str | None = None,
        expected_q_num: int | None = None,
    ) -> dict[str, Any]:
        """Sends page images to Gemini 2.5 Flash via official google-genai SDK."""
        if not image_bytes_list:
            raise ValueError("image_bytes_list cannot be empty.")

        from google.genai import types

        mime = f"image/{image_format.lower().replace('jpg', 'jpeg')}"
        parts: list[Any] = []

        for img_bytes in image_bytes_list:
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

        user_instruction = (
            f"Please transcribe this answer attempt. "
            f"Total scanned pages provided: {len(image_bytes_list)}."
        )
        if expected_q_num:
            user_instruction += f" Expected Question Number: {expected_q_num}."
        if fallback_question:
            user_instruction += f" Reference Question Text (for verification): '{fallback_question}'."

        parts.append(user_instruction)

        logger.info(
            f"Calling Vertex Vision ({self.model_id}) with {len(image_bytes_list)} page images..."
        )

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=VISION_QCAB_PARSER_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=8192,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = response.text or ""
                parsed_json = extract_json_dict(raw_text)
                if not parsed_json:
                    raise ModelInvocationError(f"Vertex Vision returned non-JSON output: {raw_text[:200]}")
                return parsed_json
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "ResourceExhausted" in err_str
                    or "quota" in err_str.lower()
                    or "rate limit" in err_str.lower()
                )
                if is_rate_limit and attempt < 4:
                    backoff = (2 ** attempt) * 2.0 + random.uniform(1.0, 3.0)
                    logger.warning(
                        "Vertex Vision OCR rate limit hit (attempt %d/5). Backing off for %.1fs...",
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                if ("client has been closed" in err_str or "10053" in err_str) and attempt < 2:
                    logger.warning(
                        "Vertex Vision connection issue (%s); recreating client and retrying attempt %d...",
                        e,
                        attempt + 1,
                    )
                    self._reset_thread_client()
                    continue

                logger.error("Error calling Vertex Vision: %s", e)
                raise ModelInvocationError(f"Vertex Vision OCR failed: {e}") from e

    @traceable(name="Vertex_Single_Page_OCR", run_type="llm", process_inputs=mask_vision_inputs)
    def parse_single_page(
        self,
        image_bytes: bytes,
        page_num: int,
        image_format: str = "jpeg",
    ) -> dict[str, Any]:
        """Parses a single page to dynamically detect question boundaries and transcribe handwritten content."""
        from google.genai import types

        mime = f"image/{image_format.lower().replace('jpg', 'jpeg')}"
        parts = [
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            f"Please transcribe Page {page_num} and detect if it starts a new question.",
        ]

        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=VISION_SINGLE_PAGE_OCR_SYSTEM_PROMPT,
                        temperature=0.1,
                        max_output_tokens=4096,
                        response_mime_type="application/json",
                    ),
                )
                raw_text = response.text or ""
                parsed_json = extract_json_dict(raw_text)
                if not parsed_json:
                    raise ModelInvocationError(
                        f"Vertex Vision returned non-JSON output for Page {page_num}: {raw_text[:200]}"
                    )
                parsed_json["page_num"] = page_num
                return parsed_json
            except Exception as e:
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "RESOURCE_EXHAUSTED" in err_str
                    or "ResourceExhausted" in err_str
                    or "quota" in err_str.lower()
                    or "rate limit" in err_str.lower()
                )
                if is_rate_limit and attempt < 4:
                    backoff = (2 ** attempt) * 1.5 + random.uniform(0.5, 1.5)
                    logger.warning(
                        "Vertex Vision single page OCR rate limit hit for Page %d (attempt %d/5). Backing off for %.1fs...",
                        page_num,
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                if ("client has been closed" in err_str or "10053" in err_str) and attempt < 2:
                    logger.warning(
                        "Vertex Vision connection issue (%s) on Page %d; retrying attempt %d...",
                        e,
                        page_num,
                        attempt + 1,
                    )
                    self._reset_thread_client()
                    continue

                logger.error("Error calling Vertex Vision on Page %d: %s", page_num, e)
                raise ModelInvocationError(f"Vertex Vision single page OCR failed on Page {page_num}: {e}") from e


def get_vision_client(
    provider: str | None = None,
    model_id: str | None = None,
) -> BaseVisionClient:
    """Returns the appropriate multimodal vision client based on active provider."""
    active_provider = (provider or settings.llm_provider).strip().lower()
    if active_provider in ("vertex", "gemini", "google"):
        return VertexVisionClient(model_id=model_id)
    return BedrockVisionClient(model_id=model_id)


