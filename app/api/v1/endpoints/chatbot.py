from fastapi import APIRouter
from pydantic import BaseModel
import google.generativeai as genai

from app.core.config import get_settings

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


class ChatRequest(BaseModel):
    history: list


@router.post("/message")
async def chatbot_message(req: ChatRequest):
    settings = get_settings()

    genai.configure(api_key=settings.gemini_api_key)

    model = genai.GenerativeModel(
        settings.gemini_model
    )

    latest = req.history[-1]["text"]

    response = model.generate_content(latest)

    return {
        "reply": response.text
    }