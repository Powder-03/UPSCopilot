"""Test script for Moonshot Kimi 2.5 on AWS Bedrock."""
from src.evaluation.model_factory import get_eval_llm


def main():
    print("Connecting to AWS Bedrock...")
    llm = get_eval_llm()
    print(f"Model: {llm.model_id} | Region: {llm.region_name}")

    prompt = "Hello! Please confirm you are active and respond in one short sentence."
    print(f"\nSending prompt: '{prompt}'")

    try:
        response = llm.invoke(prompt)
        print(f"\nResponse:\n{response.content}")
    except Exception as e:
        print(f"\nAWS Bedrock Error:\n{e}")


if __name__ == "__main__":
    main()
