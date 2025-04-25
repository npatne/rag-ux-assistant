# cspell:disable
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack.components.embedders import OpenAITextEmbedder
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack.utils import Secret
from haystack.components.builders.prompt_builder import PromptBuilder
from google import genai
from google.genai import types
import os
import requests
import logging
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("UXChatAssistant")


class UXChatAssistant:
    def __init__(self):
        logger.info("Initializing UXChatAssistant components...")
        
        # Add retry mechanism for initialization
        max_retries = 3
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                self._initialize_components()
                logger.info("All components initialized successfully.")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Initialization attempt {attempt+1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to initialize after {max_retries} attempts: {str(e)}")
                    raise RuntimeError(f"Critical initialization failure: {str(e)}")
    
    def _initialize_components(self):
        """Initialize all required components with proper error handling"""
        try:
            self.document_store = QdrantDocumentStore(
                url=os.getenv("QDRANT_URL"),
                index="NishadUXDocs",
                api_key=Secret.from_env_var("QDRANT_API_KEY"),
                embedding_dim=1536,
                use_sparse_embeddings=True
            )
            logger.info("Document store initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize document store: {str(e)}")
            raise
        
        try:
            self.dense_embedder = OpenAITextEmbedder(
                api_key=Secret.from_env_var("OPENAI_API_KEY"),
                model="text-embedding-3-small"
            )
            logger.info("Dense embedder initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize dense embedder: {str(e)}")
            raise
        
        try:
            self.sparse_embedder = FastembedSparseTextEmbedder(
                model="prithvida/Splade_PP_en_v1")
            self.sparse_embedder.warm_up()
            logger.info("Sparse embedder initialized and warmed up.")
        except Exception as e:
            logger.error(f"Failed to initialize sparse embedder: {str(e)}")
            raise
        
        try:
            self.hybrid_retriever = QdrantHybridRetriever(
                document_store=self.document_store, top_k=5)
            logger.info("Hybrid retriever initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize hybrid retriever: {str(e)}")
            raise
        
        try:
            self.ranker = FastembedRanker()
            self.ranker.warm_up()
            logger.info("Ranker initialized and warmed up.")
        except Exception as e:
            logger.error(f"Failed to initialize ranker: {str(e)}")
            raise
        
        try:
            self.llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            logger.info("LLM client initialized.")
            self.config = types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    maximum_remote_calls=28,
                    ignore_call_history=True
                ),
                safety_settings={
                    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE'
                }
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
            
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def hybrid_retrieve(self, query: str, filters: Optional[Dict] = None, top_k: int = 5) -> List[Any]:
        """Perform hybrid retrieval with error handling and retries"""
        logger.info(f"Starting hybrid retrieval for query: '{query}' with filters: {filters}")
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                # Get embeddings in parallel
                dense_future = self.executor.submit(self.dense_embedder.run, query)
                sparse_future = self.executor.submit(self.sparse_embedder.run, query)
                
                dense_embedding = dense_future.result()["embedding"]
                sparse_embedding = sparse_future.result()["sparse_embedding"]
                
                docs = self.hybrid_retriever.run(
                    query_embedding=dense_embedding,
                    query_sparse_embedding=sparse_embedding,
                    filters=filters,
                    top_k=top_k
                )["documents"]
                
                if not docs:
                    logger.warning("No documents retrieved.")
                    return []
                
                logger.info(f"Hybrid retrieval fetched {len(docs)} documents.")
                
                ranked_docs = self.ranker.run(query=query, documents=docs)["documents"]
                logger.info(f"Ranker refined documents to {len(ranked_docs)} top results.")
                return ranked_docs
                
            except Exception as e:
                if attempt < max_retries:
                    delay = 1 * (attempt + 1)
                    logger.warning(f"Retrieval attempt {attempt+1} failed: {str(e)}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All retrieval attempts failed: {str(e)}")
                    return []  # Return empty list as fallback
    
    @lru_cache(maxsize=20)
    def fetch_fallback_gif(self) -> str:
        """Fetch a fallback GIF with caching and error handling"""
        logger.info("Fetching fallback GIF...")
        api_key = os.getenv("GIPHY_API_KEY")
        default_gif = "https://media.giphy.com/media/xT9DPIlGnuHpr2yObu/giphy.gif"  # Default fallback
        
        try:
            url = f"https://api.giphy.com/v1/gifs/random?api_key={api_key}&tag=funny+fail"
            response = requests.get(url, timeout=3).json()
            gif_url = response.get("data", {}).get("images", {}).get("original", {}).get("url", default_gif)
            logger.info(f"Fallback GIF URL: {gif_url}")
            return gif_url
        except Exception as e:
            logger.warning(f"Failed to fetch GIF, using default: {str(e)}")
            return default_gif

    def fallback_response(self, retrieval_mode: str, reason: str = "No matching documents or bad question.") -> Dict:
        """Generate a fallback response"""
        gif_url = self.fetch_fallback_gif()
        return {
            "answer": f"Hmm, I couldn't quite figure that out. Maybe try rephrasing your question? Or better yet, reach out to the real Nishad at [nishadpux@gmail.com](mailto:nishadpux@gmail.com)! 📨\n\n({reason})",
            "sources": [],
            "mode": retrieval_mode,
            "fallback": True,
            "gif_url": gif_url
        }

    def build_prompt(self, documents: List, query: str, mode: str, 
                     active_case_study: Optional[str] = None, 
                     chat_context: Optional[str] = None, 
                     section_context: Optional[str] = None, 
                     summarize: bool = False) -> str:
        """Build a prompt with error handling"""
        try:
            logger.info(f"Building prompt for mode: {mode}, active_case_study: {active_case_study}, section: {section_context}, summarize: {summarize}")
            
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
            
            logger.info("Prompt constructed successfully.")
            return prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            # Fallback to a simpler prompt template
            simple_template = (
                "You are Nishad's portfolio assistant. Answer the following question based on the provided context.\n\n"
                "Context:\n" + "\n".join([doc.content for doc in documents]) + "\n\n"
                f"Question: {query}\n\n"
                "Assistant:"
            )
            return simple_template

    def get_llm_response(self, prompt: str, model: str = "gemini-2.0-flash-lite") -> str:
        """Get response from LLM with retries and error handling"""
        logger.info(f"Sending prompt to LLM using model: {model}")
        max_retries = 2
        backoff_time = 1
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.models.generate_content(
                    model=model, contents=prompt,config=self.config)
                logger.info("LLM response received successfully.")
                return response.text
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"LLM request failed (attempt {attempt+1}/{max_retries+1}): {str(e)}. Retrying in {backoff_time}s...")
                    time.sleep(backoff_time)
                    backoff_time *= 2
                else:
                    logger.error(f"All LLM attempts failed: {str(e)}")
                    return "I apologize, but I'm having trouble processing your request right now. Please try asking in a different way or contact Nishad directly at nishadpux@gmail.com."

    def get_response(self, query: str, chat_context: Optional[str] = None, 
                     active_case_study: Optional[str] = None, mode: str = "general", 
                     section_context: Optional[str] = None, summarize: bool = False) -> Dict:
        """Get response for a query with comprehensive error handling"""
        logger.info(f"Processing get_response: mode={mode}, active_case_study={active_case_study}, summarize={summarize}")
        retrieval_mode = mode
        filters = None
    
        try:
            # Input validation
            if not query or not isinstance(query, str) or len(query.strip()) == 0:
                return self.fallback_response(retrieval_mode, reason="Empty or invalid query.")
            
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
                    query = f"{query} ({section_context})"
                
                # For specific mode, retrieve more documents to ensure comprehensive coverage
                top_k = 15
            else:
                # General mode - no filters, standard retrieval
                retrieval_mode = "general"
                top_k = 5
    
            # Step 2: Primary retrieval
            logger.info(f"[🔍] Using filters: {filters}, top_k: {top_k}")
            docs = self.hybrid_retrieve(query, filters, top_k=top_k)
    
            # Step 3: Fallback check
            if not docs or len(docs) == 0:
                logger.warning("Insufficient documents retrieved — triggering fallback.")
                
                # Attempt broader retrieval if specific mode failed
                if mode == "specific" and active_case_study:
                    logger.info("Attempting broader retrieval without section filter")
                    # Try one more time without strict filtering
                    broader_filters = {
                        "field": "meta.title",
                        "operator": "contains",
                        "value": active_case_study.split()[0] if active_case_study else ""
                    }
                    docs = self.hybrid_retrieve(query, broader_filters, top_k=5)
                    
                    if not docs or len(docs) == 0:
                        return self.fallback_response(retrieval_mode, reason="Insufficient documents retrieved even with broader search.")
                else:
                    return self.fallback_response(retrieval_mode, reason="Insufficient documents retrieved.")
    
            # Step 4: Prompt & LLM response
            prompt = self.build_prompt(
                docs, query, mode, active_case_study, 
                chat_context=chat_context,
                section_context=section_context,
                summarize=summarize
            )
            
            answer = self.get_llm_response(prompt, model="gemini-2.0-flash-lite")
    
            if answer and any(p in answer.lower() for p in ["i'm sorry", "i don't have", "cannot", "unsure", "no information"]):
                # If answer looks uncertain, check if we should fallback
                uncertain_words_count = sum(1 for p in ["i'm sorry", "i don't have", "cannot", "unsure", "no information"] if p in answer.lower())
                uncertainty_ratio = uncertain_words_count / len(answer.split())
                
                if uncertainty_ratio > 0.2:  # If more than 20% of words indicate uncertainty
                    return self.fallback_response(retrieval_mode, reason="LLM expressed high uncertainty.")
    
            return {
                "answer": answer,
                "sources": [doc.meta.get("title", "Unknown Source") for doc in docs if hasattr(doc, 'meta')],
                "mode": retrieval_mode,
                "fallback": False,
                "gif_url": None,
                "steps": [
                    f"Retrieval Mode: {retrieval_mode}",
                    f"Initial Filters: {filters}"
                ]
            }
    
        except Exception as e:
            logger.error(f"[❌] Exception during retrieval: {str(e)}", exc_info=True)
            return self.fallback_response(retrieval_mode, reason=f"Internal exception occurred. Please try again.")
    
    def __del__(self):
        """Cleanup resources when the object is destroyed"""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
                logger.info("Thread executor shutdown successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")