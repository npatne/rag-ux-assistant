# api/main.py

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from rag.session_manager import session_manager
import uvicorn
import logging
import asyncio
import json
import time
import traceback
import signal
import sys
from typing import Optional, Dict, Any, List, Generator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("chat-api")

# Initialize FastAPI
app = FastAPI(
    title="Nishad's UX Chat Assistant API",
    description="API for interacting with UX Chat Assistant",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your actual frontend domain in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Request schemas with validation
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    session_id: str = Field(..., min_length=1)
    active_case_study: Optional[str] = None
    section_context: Optional[str] = None
    mode: str = "general"
    stream: bool = False  # Whether to stream the response
    
    @field_validator('mode')
    def validate_mode(cls, v):
        allowed_modes = ['general', 'specific']
        if v not in allowed_modes:
            raise ValueError(f"Mode must be one of {allowed_modes}")
        return v


class SessionInitRequest(BaseModel):
    mode: str = "general"
    active_case_study: Optional[str] = None


# Global variable for graceful shutdown
is_shutting_down = False

# Graceful shutdown handler
def handle_shutdown(signal, frame):
    global is_shutting_down
    logger.info("Shutdown signal received, closing API...")
    is_shutting_down = True
    time.sleep(2)  # Allow time for current requests to complete
    sys.exit(0)

# Register shutdown handlers
signal.signal(signal.SIGINT, handle_shutdown)
signal.signal(signal.SIGTERM, handle_shutdown)

# Health check endpoint
@app.get("/")
async def health_check():
    if is_shutting_down:
        return JSONResponse(
            status_code=503,
            content={"status": "shutting_down", "message": "Server is shutting down"}
        )
    
    # Check if session manager is properly initialized
    if not hasattr(session_manager, 'ux_assistant') or not session_manager.ux_assistant:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "message": "UX Chat API is still initializing."}
        )
    
    return {"status": "alive", "message": "Nishad's UX Chat API is running."}

@app.get("/status")
async def status_check():
    """Extended status endpoint with more details about the system"""
    try:
        # Check components and report their status
        components = {}
        
        # Check session manager
        components["session_manager"] = "ok" if hasattr(session_manager, 'ux_assistant') else "initializing"
        
        # Check active sessions
        session_count = 0
        with session_manager.sessions_lock:
            session_count = len(session_manager.sessions)
        
        return {
            "status": "alive", 
            "components": components,
            "active_sessions": session_count,
            "worker_threads": len(session_manager.workers) if hasattr(session_manager, 'workers') else 0,
        }
    except Exception as e:
        logger.error(f"Error in status check: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error checking status: {str(e)}"}
        )

# Generator for streaming response chunks
async def stream_response(response_dict: Dict[str, Any]) -> Generator[bytes, None, None]:
    """Generate streaming response chunks from the full response dictionary"""
    
    # First send metadata (everything except the answer)
    metadata = {k: v for k, v in response_dict.items() if k != "answer"}
    yield f"data: {json.dumps({'type': 'metadata', 'content': metadata})}\n\n".encode('utf-8')
    
    # If there's a fallback, send it as a complete chunk
    if response_dict.get("fallback", False):
        yield f"data: {json.dumps({'type': 'content', 'content': response_dict.get('answer', '')})}\n\n".encode('utf-8')
        yield f"data: {json.dumps({'type': 'end'})}\n\n".encode('utf-8')
        return
    
    # Get the full answer text
    answer = response_dict.get("answer", "")
    
    # Stream the answer in smaller chunks
    # Simulate streaming by sending word by word or sentence by sentence
    sentences = answer.split('. ')
    for i, sentence in enumerate(sentences):
        # Add period back except for last sentence
        if i < len(sentences) - 1:
            sentence = sentence + '.'
            
        # Stream each sentence
        yield f"data: {json.dumps({'type': 'content', 'content': sentence + ' '})}\n\n".encode('utf-8')
        await asyncio.sleep(0.1)  # Small delay between sentences
    
    # Send end of stream marker
    yield f"data: {json.dumps({'type': 'end'})}\n\n".encode('utf-8')

# Main chat endpoint with streaming support
@app.post("/chat")
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    if is_shutting_down:
        raise HTTPException(status_code=503, detail="Server is shutting down")
        
    # Validate inputs
    if not request.query.strip():
        return JSONResponse(
            status_code=400, 
            content={"error": "Query cannot be empty"}
        )
    
    try:
        # If streaming is requested
        if request.stream:
            # Process in a non-blocking way
            response_future = asyncio.get_running_loop().run_in_executor(
                None,
                lambda: session_manager.handle_query(
                    session_id=request.session_id,
                    user_query=request.query,
                    active_case_study=request.active_case_study,
                    section_context=request.section_context,
                    mode=request.mode
                )
            )
            
            # Create a wrapper coroutine to get the response and generate stream
            async def response_generator():
                try:
                    response = await response_future
                    async for chunk in stream_response(response):
                        yield chunk
                except Exception as e:
                    error_msg = {"type": "error", "content": f"Stream error: {str(e)}"}
                    yield f"data: {json.dumps(error_msg)}\n\n".encode('utf-8')
                    yield f"data: {json.dumps({'type': 'end'})}\n\n".encode('utf-8')
            
            # Return a streaming response
            return StreamingResponse(
                response_generator(),
                media_type="text/event-stream"
            )
        
        # For non-streaming requests, process normally
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
        tb = traceback.format_exc()
        logger.error(f"Full traceback: {tb}")
        
        error_response = session_manager._create_fallback_response(
            f"Internal server error occurred. Please try again later."
        )
        return error_response

@app.post("/session", tags=["session"])
async def create_session(req: SessionInitRequest):
    """
    Frontend calls this once on load to obtain a fresh session_id.
    """
    session_id = session_manager.create_new_session(
        mode=req.mode,
        active_case_study=req.active_case_study
    )
    return {"session_id": session_id}

# Custom exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Generic exception handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

# Application entry point
if __name__ == "__main__":
    try:
        # Configure uvicorn with proper shutdown handlers
        uvicorn.run(
            "api.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            workers=2,  # Adjust based on your server capacity
            log_level="info",
            timeout_keep_alive=120,  # Keep connections alive longer for streaming
            limit_concurrency=30,  # Limit concurrent connections
            limit_max_requests=5000  # Restart workers after this many requests
        )
    except Exception as e:
        logger.critical(f"Failed to start server: {e}")
        sys.exit(1)