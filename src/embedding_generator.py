from openai import OpenAI

from src.config import OPENAI_API_KEY


class EmbeddingGenerator:

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = "text-embedding-3-small"

    def generate_embedding(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )

        return response.data[0].embedding