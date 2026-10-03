"""Test script for primary evaluation LLM (Google Cloud Vertex AI Gemini or AWS Bedrock Kimi)."""
from src.config import settings
from src.providers.factory import get_eval_llm


def main():
    print(f"Active LLM Provider: {settings.llm_provider}")
    print(f"Target Model: {settings.active_eval_model_id}")
    print("Initializing model via get_eval_llm()...")

    llm = get_eval_llm()
    print(f"Model client initialized: {type(llm).__name__}")

    prompt = "Hello! Please confirm you are active and respond in one short sentence."
    print(f"\nSending prompt: '{prompt}'")

    try:
        response = llm.invoke(prompt)
        print(f"\nResponse:\n{response.content}")
        print("\n[SUCCESS] Model invocation completed successfully!")
    except Exception as e:
        print(f"\nModel Invocation Error:\n{e}")


if __name__ == "__main__":
    main()

