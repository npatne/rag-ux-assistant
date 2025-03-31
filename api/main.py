# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from rag.retriever import UXChatAssistant
import uvicorn

app = FastAPI()
assistant = UXChatAssistant()


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    result = assistant.get_response(query=req.query, session_id=req.session_id)
    return result

# Optional: local testing
if __name__ == "__main__":
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
