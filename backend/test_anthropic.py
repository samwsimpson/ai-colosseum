# backend/test_anthropic.py
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("ANTHROPIC_API_KEY is not set. Please check your .env file.")
    exit()

try:
    client = anthropic.Anthropic(api_key=api_key)

    print("Attempting a test call to the Claude API...")
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        temperature=0.7,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": "What is the capital of Texas?"}
        ]
    )
    print("\n--- Claude API Test Successful! ---")
    print("Response:", message.content[0].text)
    print("---------------------------------")

except Exception as e:
    print("\n--- Claude API Test FAILED! ---")
    print("Error:", e)
    print("-------------------------------")