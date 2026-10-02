"""Unit tests for the multi-provider architecture (Vertex AI Gemini & AWS Bedrock Kimi)."""
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.config import settings
from src.evaluation.geval_scorer import (
    BedrockGEvalScorer,
    VertexGEvalScorer,
    get_geval_scorer,
)
from src.evaluation.model_factory import get_eval_llm
from src.evaluation.vertex_chat import ChatVertexExpress
from src.models.enums import PillarType
from src.parsing.vision_client import (
    BedrockVisionClient,
    VertexVisionClient,
    get_vision_client,
)


def test_model_factory_defaults_to_vertex_chat_express():
    """When LLM_PROVIDER is vertex, get_eval_llm must return ChatVertexExpress."""
    llm = get_eval_llm(provider="vertex")
    assert isinstance(llm, ChatVertexExpress)
    assert llm.model_name == settings.vertex_eval_model_id
    assert llm.location == settings.gcp_location


def test_model_factory_selects_bedrock_when_specified():
    """When provider is bedrock, get_eval_llm should return ChatBedrockConverse."""
    with patch("boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        llm = get_eval_llm(provider="bedrock")
        assert type(llm).__name__ == "ChatBedrockConverse"


def test_geval_scorer_factory_routes_correctly():
    """get_geval_scorer returns VertexGEvalScorer or BedrockGEvalScorer accordingly."""
    vertex_scorer = get_geval_scorer(provider="vertex")
    assert isinstance(vertex_scorer, VertexGEvalScorer)
    assert vertex_scorer.model_id == settings.vertex_eval_model_id

    bedrock_scorer = get_geval_scorer(provider="bedrock")
    assert isinstance(bedrock_scorer, BedrockGEvalScorer)
    assert bedrock_scorer.model_id == settings.bedrock_eval_model_id


def test_vision_client_factory_routes_correctly():
    """get_vision_client returns VertexVisionClient or BedrockVisionClient accordingly."""
    vertex_vision = get_vision_client(provider="vertex")
    assert isinstance(vertex_vision, VertexVisionClient)
    assert vertex_vision.model_id == settings.vertex_vision_model_id

    bedrock_vision = get_vision_client(provider="bedrock")
    assert isinstance(bedrock_vision, BedrockVisionClient)
    assert bedrock_vision.model_id == settings.bedrock_vision_model_id


def test_chat_vertex_express_converts_messages():
    """ChatVertexExpress must convert SystemMessage, HumanMessage, and AIMessage to Gemini format."""
    chat = ChatVertexExpress(
        model_name="gemini-2.5-flash",
        project_id="test-proj",
        location="us-central1",
        api_key="test-key",
    )
    messages = [
        SystemMessage(content="You are an examiner."),
        HumanMessage(content="Evaluate this answer."),
        AIMessage(content="Here is the evaluation."),
        HumanMessage(content="Thank you."),
    ]
    sys_inst, contents = chat._convert_messages(messages)

    assert sys_inst == "You are an examiner."
    assert len(contents) == 3
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "Evaluate this answer."
    assert contents[1].role == "model"
    assert contents[1].parts[0].text == "Here is the evaluation."
    assert contents[2].role == "user"
    assert contents[2].parts[0].text == "Thank you."


def test_vertex_geval_scorer_parses_top_and_chosen_logprobs():
    """VertexGEvalScorer parses Gemini logprobsResult into calibrated UPSC pillar scores."""
    scorer = VertexGEvalScorer(api_key="dummy")
    output_text = "P1: 4\nP2: 3\nP3: 4\nP4: 5\nP5: 3\nP6: 4"

    # Simulated Gemini logprobsResult tokens
    content_tokens = [
        {
            "token": "P1:",
            "logprob": -0.01,
            "top_logprobs": [{"token": "P1:", "logprob": -0.01}],
        },
        {
            "token": " 4",
            "logprob": -0.356,
            "top_logprobs": [
                {"token": " 4", "logprob": -0.356},  # ~0.70
                {"token": " 5", "logprob": -1.609},  # ~0.20
                {"token": " 3", "logprob": -2.302},  # ~0.10
            ],
        },
        {
            "token": "\nP2:",
            "logprob": -0.01,
            "top_logprobs": [{"token": "\nP2:", "logprob": -0.01}],
        },
        {
            "token": " 3",
            "logprob": -0.1,
            "top_logprobs": [
                {"token": " 3", "logprob": -0.1},
            ],
        },
    ]

    pillars = scorer._parse_logprob_pillars(content_tokens, output_text, {}, 10.0)
    assert len(pillars) == 6
    p1 = pillars[PillarType.DEMAND_FULFILLMENT.value]
    assert p1.scoring_method == "logprob"
    assert 4 in p1.discrete_probabilities
    assert p1.raw_expected_rating >= 4.0
