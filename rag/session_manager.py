from datetime import datetime, timedelta
from rag.retriever import UXChatAssistant
import threading
from google import genai
import os

from dotenv import load_dotenv
load_dotenv()

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
        self.llm_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def add_chat(self, user_query, llm_answer, section_context=None):
        chat_entry = {
            "user_query": user_query,
            "llm_answer": llm_answer,
            "timestamp": datetime.now(),
            "section_context": section_context
        }
        self.chats.append(chat_entry)
        self.last_updated = datetime.now()
        self.summarize_if_needed()

    def summarize_if_needed(self):
        total_tokens = sum(len(chat["user_query"]) + len(chat["llm_answer"]) for chat in self.chats)
        if total_tokens > self.token_limit:
            chat_text = " ".join([f"User: {chat['user_query']} Assistant: {chat['llm_answer']}" for chat in self.chats])
            prompt = f"Summarize the following chat context:\n\n{chat_text}\n\nSummary:"
            summary_response = self.llm_client.models.generate_content(
                model="gemini-1.5-flash-8b", contents=prompt).text
            self.summary = summary_response
            # Retain only the last 5 chats
            self.chats = self.chats[-5:]


class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()
        self.ux_assistant = UXChatAssistant()

    def get_or_create_session(self, session_id, mode='general', active_case_study=None):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = Session(session_id, mode, active_case_study)
            return self.sessions[session_id]

    def handle_query(self, session_id, user_query, active_case_study=None, section_context=None, mode='general'):
        session = self.get_or_create_session(session_id, mode, active_case_study)

        # Update session details if needed
        session.mode = mode
        session.active_case_study = active_case_study

        # Build chat context
        chat_context = session.summary
        chat_context += " ".join([f"User: {chat['user_query']} Assistant: {chat['llm_answer']}" for chat in session.chats])

        response = self.ux_assistant.get_response(
            query=user_query,
            chat_context=chat_context,
            active_case_study=active_case_study,
            mode=mode,
            section_context=section_context
        )

        session.add_chat(user_query, response["answer"], section_context)

        return response

    def cleanup_sessions(self):
        while True:
            with self.lock:
                now = datetime.now()
                to_remove = [sid for sid, sess in self.sessions.items()
                             if now - sess.last_updated > timedelta(minutes=15)]
                for sid in to_remove:
                    del self.sessions[sid]
            threading.Event().wait(timeout=300)  # Run every 5 minutes


# Start cleanup thread
session_manager = SessionManager()
cleanup_thread = threading.Thread(target=session_manager.cleanup_sessions, daemon=True)
cleanup_thread.start()
