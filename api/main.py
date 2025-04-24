# api/main.py

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from rag.session_manager import session_manager
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chat-api")

# Initialize FastAPI
app = FastAPI(title="Nishad's UX Chat Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your actual frontend domain
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request schema
class ChatRequest(BaseModel):
    query: str
    session_id: str
    active_case_study: str | None = None
    section_context: str | None = None
    mode: str = "general"

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "alive", "message": "Nishad's UX Chat API is running."}

# Main chat endpoint
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = session_manager.handle_query(
            session_id=request.session_id,
            user_query=request.query,
            active_case_study=request.active_case_study,
            section_context=request.section_context,
            mode=request.mode
        )
        return response

    except Exception as e:
        logger.exception("An error occurred while processing the chat request.")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return {"error": exc.detail}

# Application entry point
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,  # Adjust based on your server capacity
        log_level="info"
    )
