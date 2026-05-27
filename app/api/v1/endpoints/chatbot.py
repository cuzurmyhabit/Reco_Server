import logging
from typing import List, Literal, Optional

import google.generativeai as genai
from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.services.local_chatbot import local_reply

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


BIUM_SYSTEM_PROMPT = """당신은 친근한 분리수거·재활용 도우미 챗봇 \"비움이\"입니다.

역할:
- 대한민국 분리배출 기준에 맞게 사용자에게 알기 쉬운 안내를 제공합니다.
- 플라스틱, 유리, 종이, 금속, 비닐, 일반쓰레기, 음식물 등 다양한 품목의 분리수거 방법을 설명합니다.
- 라벨 제거, 헹굼, 압착 등 구체적인 행동 단계를 안내합니다.
- 헷갈리는 품목(예: 영수증, 치킨 박스, 깨진 유리)에 대해 정확하게 안내합니다.

말투:
- 정중한 존댓말을 사용합니다.
- 짧고 명확하게 답변합니다 (3~6문장 이내).
- 필요한 경우 번호 매긴 단계로 안내합니다.
- 이모지는 사용하지 않습니다.

금지:
- 분리수거·재활용·환경·쓰레기와 무관한 주제는 \"저는 분리수거를 도와드리는 비움이예요. 그 주제는 답변하기 어려워요.\"라고 정중히 거절합니다.
- 추측이 어려울 땐 \"정확한 안내가 어려운 품목이에요. 거주 지역 분리수거 안내를 확인해 주세요.\"라고 답합니다."""


_FALLBACK_MODELS = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
)


class ChatMessage(BaseModel):
    role: Optional[Literal["user", "model"]] = "user"
    text: str


class ChatRequest(BaseModel):
    history: List[ChatMessage]


class ChatResponse(BaseModel):
    reply: str
    source: Literal["gemini", "local"] = "gemini"
    notice: Optional[str] = None


def _normalize_history(items: List[ChatMessage]) -> List[dict]:
    out = []
    for m in items:
        role = m.role if m.role in ("user", "model") else "user"
        out.append({"role": role, "text": m.text})
    return out


def _build_contents(history: List[dict]) -> List[dict]:
    return [{"role": m["role"], "parts": [{"text": m["text"]}]} for m in history]


def _try_gemini(history: List[dict], primary_model: str) -> str:
    settings = get_settings()
    genai.configure(api_key=settings.gemini_api_key)

    candidates = [primary_model] + [m for m in _FALLBACK_MODELS if m != primary_model]
    contents = _build_contents(history)

    last_exc: Optional[Exception] = None
    for model_name in candidates:
        try:
            model = genai.GenerativeModel(
                model_name,
                system_instruction=BIUM_SYSTEM_PROMPT,
            )
            response = model.generate_content(
                contents,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    top_p=0.95,
                    max_output_tokens=512,
                ),
            )
            text = (response.text or "").strip()
            if text:
                return text
            raise RuntimeError("empty response")
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            if "404" in msg or "not found" in msg:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("사용 가능한 Gemini 모델이 없습니다.")


def _notice_for_error(msg: str) -> str:
    lowered = msg.lower()
    if "403" in msg or "denied" in lowered or "permission" in lowered:
        return "Gemini 접근 권한이 차단되어 로컬 분리수거 안내로 응답 중이에요."
    if "429" in msg or "quota" in lowered:
        return "Gemini 사용량 한도에 도달해 로컬 분리수거 안내로 응답 중이에요."
    if "api key" in lowered or "expired" in lowered or "invalid_argument" in lowered:
        return "Gemini 키 문제로 로컬 분리수거 안내로 응답 중이에요."
    return "Gemini 일시 오류로 로컬 분리수거 안내로 응답 중이에요."


@router.post("/message", response_model=ChatResponse)
async def chatbot_message(req: ChatRequest) -> ChatResponse:
    settings = get_settings()
    history = _normalize_history(req.history)

    if not history:
        return ChatResponse(
            reply="안녕하세요! 분리수거 도우미 비움이예요. 어떤 품목이 궁금하신가요?",
            source="local",
            notice=None,
        )

    if not settings.gemini_enabled or not settings.gemini_api_key:
        return ChatResponse(
            reply=local_reply(history),
            source="local",
            notice="Gemini가 비활성화되어 로컬 분리수거 안내로 응답 중이에요.",
        )

    try:
        reply = _try_gemini(history, settings.gemini_model)
        return ChatResponse(reply=reply, source="gemini", notice=None)
    except Exception as exc:
        msg = str(exc)
        logger.warning("Gemini 호출 실패, 로컬 폴백으로 전환: %s", msg)
        return ChatResponse(
            reply=local_reply(history),
            source="local",
            notice=_notice_for_error(msg),
        )
