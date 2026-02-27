# main_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import core_logic
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# ---------------- APP INIT ----------------

app = FastAPI(
    title="NeurOm Chatbot API",
    description="AI-powered intelligent backend for NeurOm mental wellness assistant.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- REQUEST MODEL ----------------

class ChatQuery(BaseModel):
    query: str
    session_id: Optional[str] = None
    profile_data: Optional[dict] = None


# ---------------- CHAT ENDPOINT ----------------

@app.post("/chat", summary="Intelligent NeurOm Chat Endpoint")
async def get_chat_response(chat_query: ChatQuery):

    print("----- Incoming Request -----")
    print(f"Query: {chat_query.query}")
    print(f"Session ID: {chat_query.session_id}")

    if not core_logic.RESOURCES_INITIALIZED:
        core_logic.initialize_resources()

    if core_logic.LLM_INSTANCE is None or core_logic.RETRIEVER_INSTANCE is None:
        raise HTTPException(status_code=503, detail="Chatbot service is not ready.")

    try:
        result = core_logic.generate_llm_response(
            user_query=chat_query.query,
            session_id=chat_query.session_id,
            profile_data=chat_query.profile_data
        )

        print("----- Sending Response -----")
        print(result)

        return result

    except Exception as e:
        print(f"Error during response generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")


# ---------------- LOCAL RUN ----------------

if __name__ == "__main__":
    IS_GCF = os.getenv("K_SERVICE") is not None

    if not IS_GCF:
        print("Running locally with Uvicorn...")
        if not core_logic.RESOURCES_INITIALIZED:
            core_logic.initialize_resources()

        uvicorn.run(app, host="0.0.0.0", port=8000)
