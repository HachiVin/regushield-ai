import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = client.models.list()

print("Daftar Model Groq yang AKTIF saat ini:")
print("-" * 40)
for m in models.data:
    print(m.id)
print("-" * 40)