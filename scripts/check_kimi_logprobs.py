"""Script to check if Moonshot Kimi 2.5 on AWS Bedrock supports token logprobs."""
import json
import boto3
from botocore.exceptions import ClientError
from src.config import settings

def test_converse_api(client, model_id: str):
    """Test 1: Check standard Bedrock Converse API response structure."""
    print("=" * 60)
    print("TEST 1: Bedrock Converse API standard call")
    print("=" * 60)
    try:
        response = client.converse(
            modelId=model_id,
            messages=[
                {"role": "user", "content": [{"text": "Rate the following answer on a scale of 1 to 5: The answer is good."}]}
            ],
            inferenceConfig={"maxTokens": 50, "temperature": 0.0},
        )
        print("Response Keys:", list(response.keys()))
        output = response.get("output", {})
        print("Output:", json.dumps(output, indent=2))
        print("Usage:", response.get("usage"))
        print("Stop Reason:", response.get("stopReason"))
        print("Metrics:", response.get("metrics"))
        
        # Check if logprobs appear anywhere in the response
        has_logprobs = "logprobs" in response or "logprobs" in output
        print(f"Logprobs found in Converse response? {has_logprobs}")
    except Exception as e:
        print(f"Converse API error: {e}")

def test_converse_with_additional_fields(client, model_id: str):
    """Test 2: Request logprobs via additionalModelRequestFields in Converse."""
    print("\n" + "=" * 60)
    print("TEST 2: Converse API with additionalModelRequestFields={'logprobs': True}")
    print("=" * 60)
    try:
        response = client.converse(
            modelId=model_id,
            messages=[
                {"role": "user", "content": [{"text": "Say 'hello'"}]}
            ],
            inferenceConfig={"maxTokens": 10},
            additionalModelRequestFields={"logprobs": True, "top_logprobs": 5},
        )
        print("Success with additionalModelRequestFields!")
        print("Response keys:", list(response.keys()))
    except ClientError as e:
        print(f"Bedrock rejected additionalModelRequestFields: {e.response.get('Error', {}).get('Message', e)}")
    except Exception as e:
        print(f"Error: {e}")

def test_invoke_model_api(client, model_id: str):
    """Test 3: Check raw invoke_model endpoint with OpenAI-compatible body."""
    print("\n" + "=" * 60)
    print("TEST 3: Direct invoke_model API with G-Eval score token logprobs")
    print("=" * 60)
    body = {
        "messages": [
            {"role": "user", "content": "Rate the quality of this statement from 1 to 5 where 1 is poor and 5 is excellent. Output ONLY the single number digit (1, 2, 3, 4, or 5):\n\nStatement: 'Article 21 protects life and personal liberty.'\nRating:"}
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 5,
    }
    try:
        res = client.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        res_body = json.loads(res["body"].read().decode("utf-8"))
        print("invoke_model response keys:", list(res_body.keys()))
        first_choice = res_body["choices"][0]
        logprobs_info = first_choice.get("logprobs", {})
        content_tokens = logprobs_info.get("content", [])
        print(f"Total tokens returned with logprobs: {len(content_tokens)}")
        if content_tokens:
            first_tok = content_tokens[0]
            print("\nGenerated Token:", repr(first_tok.get("token")))
            print("Token Logprob:", first_tok.get("logprob"))
            top_lp = first_tok.get("top_logprobs", [])
            print(f"\nTop Logprobs ({len(top_lp)} candidates):")
            
            import math
            prob_dict = {}
            for item in top_lp:
                token_str = item.get("token", "").strip()
                lp = item.get("logprob", 0.0)
                prob = math.exp(lp)
                print(f"  Token: {repr(token_str):<8} | Logprob: {lp:8.4f} | Prob: {prob*100:6.2f}%")
                if token_str.isdigit():
                    prob_dict[int(token_str)] = prob
            
            if prob_dict:
                # Normalize probabilities over integer score candidates
                total_p = sum(prob_dict.values())
                expected_score = sum(score * (p / total_p) for score, p in prob_dict.items())
                print("\n" + "-" * 50)
                print(f"Normalized Probability Distribution: { {k: f'{(v/total_p)*100:.2f}%' for k, v in prob_dict.items()} }")
                print(f"G-Eval Expected Continuous Score E[S] = sum(s * P(s)): {expected_score:.3f} / 5.0")
                print("-" * 50)
    except ClientError as e:
        print(f"Bedrock invoke_model error: {e.response.get('Error', {}).get('Message', e)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"Testing AWS Bedrock model: {settings.bedrock_eval_model_id}")
    print(f"Region: {settings.aws_region}")
    
    bedrock_client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    test_invoke_model_api(bedrock_client, settings.bedrock_eval_model_id)
