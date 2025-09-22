import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # Load environment variables from .env file
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)


import os
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai

# Load API keys from .env
load_dotenv()

# === GROQ TEST ===
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": "Explain the importance of fast language models"}],
    model="llama-3.3-70b-versatile",
)
print("Groq Response:", chat_completion.choices[0].message.content)


# === GOOGLE GEMINI TEST ===
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# List available models
print("\nAvailable Google Models:")
for model in genai.list_models():
    print(" -", model.name)

# Pick one model (change if needed)
model = genai.GenerativeModel("gemini-1.5-flash")

# Ask something
response = model.generate_content("Explain the importance of fast language models")
print("\nGoogle Gemini Response:", response.text)









