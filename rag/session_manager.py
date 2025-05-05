from datetime import datetime, timedelta
from rag.retriever import UXChatAssistant
import threading
import queue
import traceback
import logging
from google import genai
from google.genai import types
import os
import time
from typing import Dict, List, Any, Optional
import uuid
import asyncio

from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SessionManager")


class Session:
    def __init__(self, session_id, mode='general', active_case_study=None):
        self.session_id = session_id
        self.mode = mode
        self.active_case_study = active_case_study
        self.created_on = datetime.now()
        self.last_updated = datetime.now()
        self.chats = []
        self.summary = ""
        self.token_limit = 500000  # Adjust based on LLM token limit
        self.summary_lock = threading.Lock()  # Prevent concurrent summarizations
        self.is_summarizing = False

        # Initialize LLM client with retry
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                self.llm_client = genai.Client(
                    api_key=os.getenv("GOOGLE_API_KEY"))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to initialize LLM client (attempt {attempt+1}): {str(e)}. Retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Failed to initialize LLM client after {max_retries} attempts")
                    self.llm_client = None  # Will trigger fallbacks when needed

    def add_chat(self, user_query, llm_answer, section_context=None):
        """Add a chat entry to the session with timestamp"""
        if not user_query or not llm_answer:
            logger.warning(
                f"Attempted to add empty chat to session {self.session_id}")
            return

        try:
            chat_entry = {
                "user_query": user_query,
                "llm_answer": llm_answer,
                "timestamp": datetime.now(),
                "section_context": section_context
            }
            self.chats.append(chat_entry)
            self.last_updated = datetime.now()

            # Trigger summarization in separate thread if needed
            total_tokens = sum(
                len(chat["user_query"]) + len(chat["llm_answer"]) for chat in self.chats)
            if total_tokens > self.token_limit and not self.is_summarizing:
                threading.Thread(target=self._summarize_chats,
                                 daemon=True).start()

        except Exception as e:
            logger.error(
                f"Error adding chat to session {self.session_id}: {str(e)}")

    def _summarize_chats(self):
        """Summarize chat history to reduce token count"""
        # Use lock to prevent multiple summarizations running concurrently
        if not self.summary_lock.acquire(blocking=False):
            logger.info(
                f"Summarization already in progress for session {self.session_id}")
            return

        try:
            self.is_summarizing = True
            logger.info(
                f"Starting summarization for session {self.session_id}")

            # Only proceed if LLM client is available
            if not self.llm_client:
                logger.warning(
                    f"Cannot summarize session {self.session_id}: LLM client unavailable")
                return

            # Build context from chats
            chat_text = " ".join([f"User: {chat.get('user_query', '')} Assistant: {chat.get('llm_answer', '')}"
                                  for chat in self.chats if chat.get('user_query') and chat.get('llm_answer')])

            if not chat_text:
                logger.warning(
                    f"No valid chat text to summarize for session {self.session_id}")
                return

            prompt = f"Summarize the following chat context concisely. Capture key topics and information exchanged:\n\n{chat_text}\n\nSummary:"

            # Handle potential LLM errors
            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-1.5-flash-8b", contents=prompt, config=types.GenerateContentConfig(
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(
                            maximum_remote_calls=14,
                            ignore_call_history=True
                        )
                    ))
                new_summary = response.text

                if new_summary:
                    self.summary = new_summary
                    # Keep only the most recent conversations
                    self.chats = self.chats[-5:]
                    logger.info(
                        f"Successfully summarized session {self.session_id}, kept last 5 chats")
                else:
                    logger.warning(
                        f"Empty summary returned for session {self.session_id}")
            except Exception as e:
                logger.error(
                    f"LLM summarization failed for session {self.session_id}: {str(e)}")

        except Exception as e:
            logger.error(
                f"Error during chat summarization for session {self.session_id}: {str(e)}")
            traceback.print_exc()
        finally:
            self.is_summarizing = False
            self.summary_lock.release()

    def get_context(self) -> str:
        """Get the current context for this session"""
        try:
            context = self.summary
            # Add recent chats that aren't part of the summary
            chat_context = " ".join([
                f"User: {chat.get('user_query', '')} Assistant: {chat.get('llm_answer', '')}"
                for chat in self.chats if chat.get('user_query') and chat.get('llm_answer')
            ])
            return f"{context} {chat_context}".strip()
        except Exception as e:
            logger.error(
                f"Error getting context for session {self.session_id}: {str(e)}")
            return ""


class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.sessions_lock = threading.RLock()  # Reentrant lock for thread safety
        self.request_queue = queue.Queue()
        self.max_workers = 4
        self.workers = []
        self.running = True
        self.initialization_event = threading.Event()

        # Initialize UXChatAssistant in a separate thread
        init_thread = threading.Thread(target=self._initialize_assistant)
        init_thread.daemon = True
        init_thread.start()

        # Wait for initialization with timeout
        if not self.initialization_event.wait(timeout=30):
            logger.warning(
                "UXChatAssistant initialization is taking longer than expected")

    def _initialize_assistant(self):
        """Initialize the UX Assistant with retries"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info("Initializing UXChatAssistant...")
                self.ux_assistant = UXChatAssistant()

                # Start worker threads for handling requests
                for _ in range(self.max_workers):
                    worker = threading.Thread(
                        target=self._process_request_worker, daemon=True)
                    worker.start()
                    self.workers.append(worker)

                # Start cleanup thread
                cleanup_thread = threading.Thread(
                    target=self.cleanup_sessions, daemon=True)
                cleanup_thread.start()

                self.initialization_event.set()
                logger.info("Session manager fully initialized.")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to initialize UXChatAssistant (attempt {attempt+1}): {str(e)}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(
                        f"Failed to initialize UXChatAssistant after {max_retries} attempts: {str(e)}")
                    # Create a dummy fallback instance
                    self.ux_assistant = None
                    self.initialization_event.set()  # Allow system to continue with fallbacks

    def get_or_create_session(self, session_id, mode='general', active_case_study=None):
        """Get existing session or create a new one"""
        # Generate random session ID if none provided
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.info(f"Generated new session ID: {session_id}")

        with self.sessions_lock:
            if session_id not in self.sessions:
                logger.info(
                    f"Creating new session: {session_id}, mode: {mode}, case study: {active_case_study}")
                self.sessions[session_id] = Session(
                    session_id, mode, active_case_study)
            return self.sessions[session_id]
 
    def create_new_session(self,
                       mode: str = "general",
                       active_case_study: str | None = None) -> str:
        """
        Always make a brand-new Session object and return its UUID.
        Re-uses get_or_create_session() so we don’t duplicate locking logic.
        """
        # Passing None forces get_or_create_session() to cook up a new uuid
        session = self.get_or_create_session(
            session_id=None,
            mode=mode,
            active_case_study=active_case_study
        )
        return session.session_id

    def _create_fallback_response(self, message):
        """Create a standardized fallback response"""
        return {
            "answer": message,
            "sources": [],
            "mode": "fallback",
            "fallback": True,
            "gif_url": None
        }

    def handle_query(self, session_id, user_query, active_case_study=None, section_context=None, mode='general'):
        """Handle a query by putting it in the queue and returning a response"""
        try:
            if not self.initialization_event.is_set():
                logger.warning(
                    "Session manager not fully initialized yet, waiting...")
                if not self.initialization_event.wait(timeout=10):
                    return self._create_fallback_response("System is still initializing. Please try again in a moment.")

            if not self.ux_assistant:
                return self._create_fallback_response("Chat assistant is not available at the moment. Please try again later.")

            # Validate inputs
            if not user_query or not isinstance(user_query, str):
                return self._create_fallback_response("Empty or invalid query received.")

            # Get or create session
            session = self.get_or_create_session(
                session_id, mode, active_case_study)

            # Update session details
            with self.sessions_lock:
                session.mode = mode
                session.active_case_study = active_case_study

            # Get chat context
            chat_context = session.get_context()

            # Process the request
            try:
                response = self.ux_assistant.get_response(
                    query=user_query,
                    chat_context=chat_context,
                    active_case_study=active_case_study,
                    mode=mode,
                    section_context=section_context
                )

                # Only add to chat history if successful
                if response and not response.get("fallback", False):
                    session.add_chat(
                        user_query, response["answer"], section_context)

                return response

            except Exception as e:
                logger.error(
                    f"Error processing query for session {session_id}: {str(e)}")
                return self._create_fallback_response(f"Error processing your request: {str(e)}")

        except Exception as e:
            logger.error(f"Unhandled exception in handle_query: {str(e)}")
            return self._create_fallback_response("An unexpected error occurred. Please try again.")

    def _process_request_worker(self):
        """Worker thread to process requests from the queue"""
        while self.running:
            try:
                # Get request from queue with timeout
                request = self.request_queue.get(timeout=1)

                # Process request
                session_id = request.get("session_id")
                user_query = request.get("user_query")
                callback = request.get("callback")

                try:
                    response = self.handle_query(
                        session_id=session_id,
                        user_query=user_query,
                        active_case_study=request.get("active_case_study"),
                        section_context=request.get("section_context"),
                        mode=request.get("mode", "general")
                    )

                    # Execute callback if provided
                    if callback:
                        callback(response)

                except Exception as e:
                    logger.error(
                        f"Error in worker processing request: {str(e)}")
                    if callback:
                        callback(self._create_fallback_response(
                            f"Processing error: {str(e)}"))

                finally:
                    self.request_queue.task_done()

            except queue.Empty:
                # Queue timeout, just continue
                pass
            except Exception as e:
                logger.error(f"Unhandled error in worker thread: {str(e)}")
                # Prevent CPU spike if there's a persistent error
                time.sleep(1)

    def enqueue_request(self, session_id, user_query, callback=None, **kwargs):
        """Add a request to the processing queue"""
        request = {
            "session_id": session_id,
            "user_query": user_query,
            "callback": callback,
            **kwargs
        }
        self.request_queue.put(request)
        return True

    def cleanup_sessions(self):
        """Periodically clean up inactive sessions"""
        cleanup_interval = 60*10  # 10 minutes

        while self.running:
            try:
                logger.debug("Running session cleanup")
                with self.sessions_lock:
                    now = datetime.now()
                    # Find sessions to remove - inactive for more than 15 minutes
                    to_remove = [
                        sid for sid, sess in self.sessions.items()
                        if now - sess.last_updated > timedelta(minutes=15)
                    ]

                    # Log cleanup activity
                    if to_remove:
                        logger.info(
                            f"Cleaning up {len(to_remove)} inactive sessions")

                    # Remove expired sessions
                    for sid in to_remove:
                        try:
                            del self.sessions[sid]
                        except KeyError:
                            pass  # Already removed

            except Exception as e:
                logger.error(f"Error during session cleanup: {str(e)}")

            # Sleep until next cleanup cycle
            time.sleep(cleanup_interval)

    def shutdown(self):
        """Gracefully shut down the session manager"""
        logger.info("Shutting down SessionManager...")
        self.running = False

        # Wait for workers to finish
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2)

        # Clear resources
        with self.sessions_lock:
            self.sessions.clear()

        logger.info("SessionManager shutdown complete")

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.shutdown()
        except:
            pass


# Initialize session manager singleton
try:
    session_manager = SessionManager()
    logger.info("SessionManager singleton initialized")

    # No need to explicitly start cleanup thread as it's handled in the constructor
except Exception as e:
    logger.critical(f"Failed to initialize SessionManager: {str(e)}")
    # Create minimal session manager that will return fallbacks
    session_manager = SessionManager()
