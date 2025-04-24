# cspell:disable
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from google import genai
import os
import random
import requests

from dotenv import load_dotenv
load_dotenv()


class UXChatAssistant:
    def __init__(self):
        print("Initializing UXChatAssistant components...")
        self.document_store = QdrantDocumentStore(
            url=os.getenv("QDRANT_URL"),
            index="NishadUXDocs",
            api_key=Secret.from_env_var("QDRANT_API_KEY"),
            embedding_dim=1536,
            use_sparse_embeddings=True
        )
        print("Document store initialized.")

        self.dense_embedder = OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model="text-embedding-3-small"
        )
        print("Dense embedder initialized.")

        self.sparse_embedder = FastembedSparseTextEmbedder(
            model="prithvida/Splade_PP_en_v1")
        self.sparse_embedder.warm_up()
        print("Sparse embedder initialized and warmed up.")

        self.hybrid_retriever = QdrantHybridRetriever(
            document_store=self.document_store, top_k=5)
        print("Hybrid retriever initialized.")

        self.ranker = FastembedRanker()
        self.ranker.warm_up()
        print("Ranker initialized and warmed up.")

        self.llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        print("LLM client initialized.")

    def hybrid_retrieve(self, query, filters=None, top_k=5):
        print(
            f"Starting hybrid retrieval for query: '{query}' with filters: {filters}")
        dense_embedding = self.dense_embedder.run(query)["embedding"]
        sparse_embedding = self.sparse_embedder.run(query)["sparse_embedding"]

        docs = self.hybrid_retriever.run(
            query_embedding=dense_embedding,
            query_sparse_embedding=sparse_embedding,
            filters=filters,
            top_k=top_k
        )["documents"]
        print(f"Hybrid retrieval fetched {len(docs)} documents.")

        ranked_docs = self.ranker.run(query=query, documents=docs)["documents"]
        print(f"Ranker refined documents to {len(ranked_docs)} top results.")
        return ranked_docs

    def fetch_fallback_gif(self):
        print("Fetching fallback GIF...")
        api_key = os.getenv("GIPHY_API_KEY")
        url = f"https://api.giphy.com/v1/gifs/random?api_key={api_key}&tag=funny+fail"
        response = requests.get(url).json()
        gif_url = response["data"]["images"]["original"]["url"]
        print(f"Fallback GIF URL: {gif_url}")
        return gif_url

    def fallback_response(self, retrieval_mode, reason="No matching documents or bad question."):
        gif_url = self.fetch_fallback_gif()
        return {
            "answer": f"Hmm, I couldn't quite figure that out. Maybe try rephrasing your question? Or better yet, reach out to the real Nishad at [nishadpux@gmail.com](mailto:nishadpux@gmail.com)! 📨\n\n({reason})",
            "sources": [],
            "mode": retrieval_mode,
            "fallback": True,
            "gif_url": gif_url
        }

    def build_prompt(self, documents, query, mode, active_case_study=None, chat_context=None, section_context=None, summarize=False):
        print(f"Building prompt for mode: {mode}, active_case_study: {active_case_study}, section: {section_context}, summarize: {summarize}")
        
        # Base structure
        system_instruction = ""
        guidelines = [
            "- Assume the persona of Nishad's helpful portfolio assistant.",
            "- Be conversational and engaging.",
            "- Keep responses concise, ideally under 150 words.",
            "- Use the provided context (documents and chat history) to answer accurately.",
            "- If chat history exists, maintain the flow of the conversation.",
            "- Encourage further exploration of Nishad's work."
        ]
        context_section = (
            "Context Documents:\n"
            "{% for doc in documents %}\n"
            "{{ doc.content }}\n"
            "{% endfor %}\n\n"
            "Chat History:\n{{chat_context}}"
        )
        question_section = "User Question:\n{{question}}"
        answer_format = "Assistant:"

        # Mode-specific adjustments
        if mode == "general":
            system_instruction = "You are Nishad's portfolio assistant, helping visitors explore Nishad's general UX/UI work and professional experience."
            guidelines.extend([
                "- Provide clear, focused answers based on the general context.",
                "- If discussing projects briefly, highlight key design decisions and outcomes."
            ])
        elif mode == "specific":
            if section_context:
                system_instruction = f"You are Nishad's portfolio assistant, discussing the '{active_case_study}' case study, specifically focusing on the '{section_context}' section."
                guidelines.extend([
                    f"- Focus on the '{section_context}' section, using the provided case study content.",
                    "- Highlight key decisions, challenges, and outcomes relevant to this section.",
                    "- Reference specific examples from the documentation for this section."
                ])
                context_section = (
                    f"Case Study Content ({active_case_study} - {section_context}):\n"
                    "{% for doc in documents %}\n"
                    "{{ doc.content }}\n"
                    "{% endfor %}\n\n"
                    "Chat History:\n{{chat_context}}"
                )
            elif summarize:
                system_instruction = f"You are Nishad's portfolio assistant, creating an engaging summary of the '{active_case_study}' case study."
                guidelines.extend([
                    "- Provide a comprehensive overview capturing the project's essence.",
                    "- Present the project's context, challenges, and key outcomes concisely.",
                    "- Highlight significant design decisions and their impact.",
                    "- Structure the summary to encourage deeper exploration."
                ])
                context_section = (
                    f"Case Study Content ({active_case_study}):\n"
                    "{% for doc in documents %}\n"
                    "{{ doc.content }}\n"
                    "{% endfor %}\n\n"
                    "Chat History:\n{{chat_context}}"
                )
                answer_format = "Summary:"
            else: # Default specific mode (discussing a case study generally)
                system_instruction = f"You are Nishad's portfolio assistant, discussing the '{active_case_study}' case study in detail."
                guidelines.extend([
                    f"- Focus on relevant aspects of the '{active_case_study}' document.",
                    "- Provide specific examples and outcomes from the document content.",
                    "- Connect different aspects of the project when relevant."
                ])
                context_section = (
                    f"Document Content ({active_case_study}):\n"
                    "{% for doc in documents %}\n"
                    "{{ doc.content }}\n"
                    "{% endfor %}\n\n"
                    "Chat History:\n{{chat_context}}"
                )

        # Assemble the final template
        template = (
            f"{system_instruction}\n\n"
            "Guidelines:\n"
            f"{'\n'.join(guidelines)}\n\n"
            f"{context_section}\n\n"
            f"{question_section}\n\n"
            f"{answer_format}"
        )

        prompt_builder = PromptBuilder(template=template)
        prompt = prompt_builder.run(
            documents=documents,
            question=query,
            chat_context=chat_context or "No previous conversation."
        )["prompt"]
        # print("\n--- Constructed Prompt ---\n", prompt, "\n-------------------------") # Optional: Uncomment to debug prompt
        print("Prompt constructed successfully.")
        return prompt

    def get_llm_response(self, prompt, model="gemini-2.0-flash-lite"):
        print("Sending prompt to LLM...")
        # gemini-1.5-flash-8b for summarization of chat context if it exceeds a set number of tokens. 
        # gemini-2.0-flash-lite for general and specific retrieval modes.
        response = self.llm_client.models.generate_content(
            model=model, contents=prompt)
        print("LLM response received.")
        return response.text

    def get_response(self, query, chat_context=None, active_case_study=None,
                     mode="general", section_context=None, summarize=False):
        print(
            f"Processing get_response: mode={mode}, active_case_study={active_case_study}, summarize={summarize}")
        retrieval_mode = mode
        filters = None
    
        try:
            # Step 1: Define filters based on mode
            if mode == "specific" and active_case_study:
                # For specific mode, always filter by case study title
                filters = {
                    "field": "meta.title",
                    "operator": "==",
                    "value": active_case_study
                }
                
                # Add section context to query for better sparse matching
                if section_context:
                    query += f" ({section_context})"
                
                # For specific mode, retrieve more documents to ensure comprehensive coverage
                top_k = 15
    
            else:
                # General mode - no filters, standard retrieval
                retrieval_mode = "general"
                top_k = 5
    
            # Step 2: Primary retrieval
            print(f"[🔍] Using filters: {filters}, top_k: {top_k}")
            docs = self.hybrid_retrieve(query, filters, top_k=top_k)
    
            # Step 3: Fallback check
            if not docs or len(docs) == 0:
                print("Insufficient documents retrieved — triggering fallback.")
                return self.fallback_response(retrieval_mode, reason="Insufficient documents retrieved.")
    
            # Step 4: Prompt & LLM response
            prompt = self.build_prompt(
                docs, query, mode, active_case_study, 
                chat_context=chat_context,
                section_context=section_context,
                summarize=summarize
            )
            
            # Always use gemini-2.0-flash-lite
            answer = self.get_llm_response(prompt, model="gemini-2.0-flash-lite")
    
            if any(p in answer.lower() for p in ["i'm sorry", "i don't have", "cannot", "unsure", "no information"]):
                return self.fallback_response(retrieval_mode, reason="LLM expressed uncertainty.")
    
            return {
                "answer": answer,
                "sources": [doc.meta.get("title") for doc in docs],
                "mode": retrieval_mode,
                "fallback": False,
                "gif_url": None,
                "steps": [
                    f"Retrieval Mode: {retrieval_mode}",
                    f"Initial Filters: {filters}"
                ]
            }
    
        except Exception as e:
            print(f"[❌] Exception during retrieval: {e}")
            return self.fallback_response(retrieval_mode, reason=f"Internal exception: {e}")
