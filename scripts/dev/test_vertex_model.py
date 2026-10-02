import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
project_id = "project-1b52589d-0827-46ab-9be"

candidate_models = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-lite-preview-02-05",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.7-flash",
]

def main():
    if not api_key:
        print("Error: GEMINI_API_KEY is not set.")
        return

    print(f"Testing Vertex AI Model Garden endpoints for project: {project_id}\n")
    success = False

    for model in candidate_models:
        print(f"--> Trying model: {model}...")
        url = f"https://us-central1-aiplatform.googleapis.com/v1/projects/{project_id}/locations/us-central1/publishers/google/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello! Confirm you are active in one short sentence."}]}
            ]
        }
        try:
            resp = requests.post(url, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                reply = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                print(f"\n[SUCCESS] Model '{model}' responded successfully!")
                print(f"Reply: {reply}\n")
                success = True
                break
            else:
                err_msg = resp.json().get("error", {}).get("message", resp.text[:200])
                print(f"    Failed ({resp.status_code}): {err_msg}")
        except Exception as e:
            print(f"    Error: {e}")

    if not success:
        print("\nNone of the direct publisher endpoints responded on us-central1 with this key.")

if __name__ == "__main__":
    main()
