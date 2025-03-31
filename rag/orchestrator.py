from google import genai
import os
from qdrant_client import QdrantClient

from openai import OpenAI

openai_client = OpenAI()
oa_response = openai_client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)

print("OPEN AI", oa_response.data[0].embedding)


qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

print("QDrant", qdrant_client.get_collections())

google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
ga_response = google_client.models.generate_content(
    model="gemini-2.0-flash-lite", contents="Explain how AI works in a few words"
)
print("Google AI", ga_response.text)
