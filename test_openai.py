import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

try:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize the following: Cats are animals that like to sleep."}
        ],
        max_tokens=50,
        temperature=0.5
    )
    print("✅ Success:\n", response.choices[0].message.content.strip())
except Exception as e:
    print("❌ Error:\n", e)
