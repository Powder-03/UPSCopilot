"""LangChain-compatible Chat Model for Google Cloud Vertex AI using official google-genai SDK."""
import logging
import random
import time
from typing import Any

from google import genai
from google.genai import types
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from src.models.exceptions import ModelInvocationError

logger = logging.getLogger(__name__)


class ChatVertexExpress(BaseChatModel):
    """
    ChatModel implementation connecting to Google Cloud Vertex AI
    using the official google-genai SDK and an API key.
    """

    model_name: str = Field(default="gemini-2.5-flash")
    project_id: str = Field(...)
    location: str = Field(default="us-central1")
    api_key: str = Field(...)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=8192)
    timeout: int = Field(default=60)

    _cached_client: Any = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "vertex-express-chat"

    @property
    def model_id(self) -> str:
        return self.model_name

    def _get_client(self) -> genai.Client:
        if self._cached_client is None:
            self._cached_client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
                api_key=self.api_key,
            )
        return self._cached_client

    def _convert_messages(self, messages: list[BaseMessage]) -> tuple[str | None, list[types.Content]]:
        """Converts LangChain messages to Gemini systemInstruction and Content objects."""
        system_texts: list[str] = []
        contents: list[types.Content] = []

        for msg in messages:
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if isinstance(msg, SystemMessage):
                system_texts.append(text)
            elif isinstance(msg, AIMessage):
                contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
            else:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))

        system_instruction = "\n\n".join(system_texts) if system_texts else None
        if not contents:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

        return system_instruction, contents

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Invokes the Vertex AI Gemini generate_content endpoint via google-genai SDK."""
        system_instruction, contents = self._convert_messages(messages)
        client = self._get_client()

        config = types.GenerateContentConfig(
            temperature=kwargs.get("temperature", self.temperature),
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            stop_sequences=stop,
            system_instruction=system_instruction,
            response_mime_type=kwargs.get("response_mime_type"),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

        for attempt in range(5):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                reply_text = response.text or ""

                # Extract token usage metadata for LangSmith cost & token tracking
                usage_metadata = None
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    meta = response.usage_metadata
                    input_toks = getattr(meta, "prompt_token_count", 0) or 0
                    output_toks = getattr(meta, "candidates_token_count", 0) or 0
                    total_toks = getattr(meta, "total_token_count", 0) or (input_toks + output_toks)
                    usage_metadata = {
                        "input_tokens": input_toks,
                        "output_tokens": output_toks,
                        "total_tokens": total_toks,
                    }

                ai_message = AIMessage(
                    content=reply_text,
                    usage_metadata=usage_metadata,
                    response_metadata={
                        "model_name": self.model_name,
                        "usage": usage_metadata or {},
                    },
                )
                return ChatResult(generations=[ChatGeneration(message=ai_message)])
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
                        "Vertex AI rate limit hit (attempt %d/5). Backing off for %.1fs...",
                        attempt + 1,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                if (
                    "closed" in err_str
                    or "10053" in err_str
                    or "connection" in err_str.lower()
                    or "aborted" in err_str.lower()
                ) and attempt < 2:
                    logger.warning(
                        "Vertex AI Gemini connection error (%s); retrying attempt %d...",
                        e,
                        attempt + 1,
                    )
                    self._cached_client = None
                    client = self._get_client()
                    continue
                logger.error("Error calling Vertex AI (%s): %s", self.model_name, e)
                raise ModelInvocationError(
                    f"Vertex AI Gemini invocation failed: {e}. Please verify your API key and connection."
                ) from e
