"""LangChain-compatible Chat Model for Google Cloud Vertex AI using official google-genai SDK."""
import logging
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
from pydantic import Field

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

    @property
    def _llm_type(self) -> str:
        return "vertex-express-chat"

    @property
    def model_id(self) -> str:
        return self.model_name

    def _get_client(self) -> genai.Client:
        return genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
            api_key=self.api_key,
        )

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

        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                reply_text = response.text or ""
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply_text))])
            except Exception as e:
                err_str = str(e)
                if (
                    "closed" in err_str
                    or "10053" in err_str
                    or "connection" in err_str.lower()
                    or "aborted" in err_str.lower()
                ) and attempt < 2:
                    logger.warning(
                        f"Vertex AI Gemini connection error ({e}); retrying attempt {attempt + 1}..."
                    )
                    client = self._get_client()
                    continue
                logger.error(f"Error calling Vertex AI ({self.model_name}): {e}")
                raise ModelInvocationError(
                    f"Vertex AI Gemini invocation failed: {e}. Please verify your API key and connection."
                ) from e
