import os
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()


class QdrantManager:
    def __init__(self):
        self.document_store = QdrantDocumentStore(
            url=os.getenv("QDRANT_URL"),
            index="NishadUXDocs",
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            embedding_dim=1536,
            use_sparse_embeddings=True,   # ✅ Enables sparse vector field
            return_embedding=True,
            recreate_index=False
        )

        self.dense_embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small",
            meta_fields_to_embed=[
                "title", "category", "keywords", "description"]
        )

        self.sparse_embedder = FastembedSparseDocumentEmbedder(
            model="prithvida/Splade_PP_en_v1",
            meta_fields_to_embed=[
                "title", "category", "keywords", "description"]
        )
        self.sparse_embedder.warm_up()

        self.writer = DocumentWriter(self.document_store)

    def embed_and_upload(self, chunks):
        print("🔍 Generating sparse embeddings...")
        chunks = self.sparse_embedder.run(chunks)['documents']

        print("🧠 Generating dense embeddings...")
        chunks = self.dense_embedder.run(chunks)['documents']

        print("📦 Uploading documents with dense + sparse embeddings to Qdrant...")
        self.writer.run(chunks)

    def inspect_embeddings(self, limit=5, show_vector_preview=False):
        docs = self.document_store.filter_documents()
        print(f"\n✅ Total documents in Qdrant: {len(docs)}\n")
        for i, doc in enumerate(docs[:limit]):
            print(f"--- Chunk {i + 1} ---")
            print("Content:", doc.content[:300].replace("\n", " "), "...\n")
            print("Metadata:", doc.meta)

            if doc.embedding:
                print(
                    f"✅ Dense Embedding exists. Length: {len(doc.embedding)}")
                if show_vector_preview:
                    print("🔍 Dense Vector preview (first 5 dims):",
                          doc.embedding[:5])
            else:
                print("❌ No dense embedding found!")

            if hasattr(doc, "sparse_embedding") and doc.sparse_embedding:
                print(
                    f"✅ Sparse embedding present with {len(doc.sparse_embedding.indices)} non-zero terms.")
            else:
                print("❌ No sparse embedding found!")

            print("-" * 60)
