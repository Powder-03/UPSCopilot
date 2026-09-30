"""Multimodal Bedrock Vision Client for extracting handwritten text, diagrams, and QCAB layouts."""
import logging
from typing import Any

import boto3

from src.config import settings
from src.parsing.prompts import VISION_QCAB_PARSER_SYSTEM_PROMPT
from src.utils.json import extract_json_dict

logger = logging.getLogger(__name__)


class BedrockVisionClient:
    """Invokes AWS Bedrock Multimodal Vision models via Converse API."""

    def __init__(self, model_id: str | None = None, region_name: str | None = None):
        self.region_name = region_name or settings.aws_region
        self.model_id = model_id or settings.bedrock_vision_model_id
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def extract_answer_from_page_images(
        self,
        image_bytes_list: list[bytes],
        image_format: str = "jpeg",
        fallback_question: str | None = None,
        expected_q_num: int | None = None,
    ) -> dict[str, Any]:
        """
        Sends 1 or more page images (e.g. all pages belonging to one question) to Bedrock Vision
        and extracts the question text, handwritten candidate answer, and diagram notations.
        """
        if not image_bytes_list:
            raise ValueError("image_bytes_list cannot be empty.")

        # Build multimodal content array with all pages for this question
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
            raise
