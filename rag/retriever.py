from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from google import genai
import os

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class UXChatAssistant:
    def __init__(self):
        print("Initializing Document Store, Embedders, Hybrid Retriever, and Ranker...")
        self.document_store = QdrantDocumentStore(
            url="https://a8ca5037-dc8a-4881-8918-854da592c1f4.europe-west3-0.gcp.cloud.qdrant.io:6333",
            index="NishadUXDocs",
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            embedding_dim=1536,
            use_sparse_embeddings=True
        )

        self.dense_embedder = OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )

        self.sparse_embedder = FastembedSparseTextEmbedder(
            model="prithvida/Splade_PP_en_v1")
        self.sparse_embedder.warm_up()

        self.hybrid_retriever = QdrantHybridRetriever(
            document_store=self.document_store, top_k=5)

        self.ranker = FastembedRanker()
        self.ranker.warm_up()

        self.prompt_template = """
        You're an expert UX assistant answering questions strictly based on provided documents.  
        If no answer is found, politely mention it.

        Context:
        {% for doc in documents %}
        Document: {{ doc.meta.title }}
        {{ doc.content }}
        {% endfor %}

        Question: {{question}}

        Answer:
        """
        self.prompt_builder = PromptBuilder(template=self.prompt_template)

        self.llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.sessions = {}
        print("Setup Complete!")

    def dynamic_metadata_filter(self, query):
        query_lower = query.lower()
        conditions = []
        if "case study" in query_lower:
            conditions.append(
                {"field": "meta.doc_type", "operator": "==", "value": "case_study"})
        if any(keyword in query_lower for keyword in ["resume", "experience", "job", "role"]):
            conditions.append(
                {"field": "meta.doc_type", "operator": "==", "value": "resume"})
        if any(keyword in query_lower for keyword in ["project", "contribution"]):
            conditions.append(
                {"field": "meta.doc_type", "operator": "==", "value": "contribution"})

        return {"operator": "AND", "conditions": conditions} if conditions else None

    def hybrid_retrieve(self, query, filters=None):
        print(
            f"Performing hybrid retrieval for query: '{query}' with filters: {filters}")

        dense_embedding = self.dense_embedder.run(query)["embedding"]
        sparse_embedding = self.sparse_embedder.run(query)["sparse_embedding"]

        docs = self.hybrid_retriever.run(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            # filters=filters will impliment this later as needed.
        )["documents"]

        print(f"Retrieved {len(docs)} documents, ranking now...")
        ranked_docs = self.ranker.run(query=query, documents=docs)["documents"]
        print(f"Ranked documents obtained: {len(ranked_docs)}")
        return ranked_docs

    def build_prompt(self, documents, query):
        prompt = self.prompt_builder.run(documents=documents, question=query)
        print("Prompt built successfully.")
        return prompt["prompt"]

    def get_llm_response(self, prompt):
        response = self.llm_client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )
        return response.text

    def chat(self):
        print("Welcome to UX Document Chat! Type 'exit' to quit.")
        while True:
            query = input("\nAsk your UX question: ")
            if query.lower() == 'exit':
                print("Goodbye!")
                break

            # filters = self.dynamic_metadata_filter(query)
            # add filter argument here later.
            ranked_documents = self.hybrid_retrieve(query)
            prompt = self.build_prompt(ranked_documents, query)
            answer = self.get_llm_response(prompt)

            print("\nAnswer from Gemini:")
            print(answer)

    def get_response(self, query, session_id="default"):
        print(f"🟢 Session: {session_id} | Incoming query: {query}")
        steps = []

        try:
            steps.append("🔍 Retrieving documents...")
            # filters = self.dynamic_metadata_filter(query)
            docs = self.hybrid_retrieve(query)

            steps.append("📄 Documents retrieved. Building prompt...")
            prompt = self.build_prompt(docs, query)

            steps.append("🤖 Sending to LLM...")
            answer = self.get_llm_response(prompt)

            steps.append("✅ Answer received.")

            # Save to session history
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append({
                "question": query,
                "answer": answer
            })

            return {
                "answer": answer,
                "sources": [doc.meta.get("title", "") for doc in docs],
                "steps": steps
            }

        except Exception as e:
            steps.append("❌ Error occurred.")
            return {
                "error": str(e),
                "steps": steps
            }


if __name__ == "__main__":
    assistant = UXChatAssistant()
    assistant.chat()
