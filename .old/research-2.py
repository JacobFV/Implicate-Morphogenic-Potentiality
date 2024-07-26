import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGAgent:
    ANTHROPIC_MODEL = "claude-2.1"
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    MAX_TOKENS = 300
    INITIAL_KNOWLEDGE = [
        "The capital of France is Paris.",
        "Python is a popular programming language.",
        "The Earth orbits around the Sun.",
    ]

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.embeddings_model = SentenceTransformer(self.SENTENCE_TRANSFORMER_MODEL)
        self.knowledge_base = self.INITIAL_KNOWLEDGE
        self.knowledge_embeddings = self.embeddings_model.encode(self.knowledge_base)

    def update_knowledge_base(self, new_info):
        self.knowledge_base.append(new_info)
        new_embedding = self.embeddings_model.encode([new_info])[0]
        self.knowledge_embeddings = np.vstack(
            [self.knowledge_embeddings, new_embedding]
        )

    def generate_response(self, user_input):
        query_embedding = self.embeddings_model.encode([user_input])[0]
        similarities = cosine_similarity([query_embedding], self.knowledge_embeddings)[
            0
        ]
        top_indices = np.argsort(similarities)[-3:][::-1]  # Get top 3 most similar
        context = "\n".join([self.knowledge_base[i] for i in top_indices])

        prompt = f"""Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {user_input}
        Answer:"""

        response = self.client.completions.create(
            model=self.ANTHROPIC_MODEL,
            max_tokens_to_sample=self.MAX_TOKENS,
            prompt=prompt,
        )
        return response.completion
