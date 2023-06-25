import datetime
import json
from logging import Logger

import arrow
from fastapi import APIRouter
from fastapi import Request

from api.model.llm import ChatGLMRequest, QA, ChatGLMResponse

router = APIRouter(prefix="/api/llm", tags=["LLM"])


@router.post("/chat", summary="Chat with ChatGLM model.", response_model=ChatGLMResponse)
async def send_chat_request(req: Request, chat_request: ChatGLMRequest):
    logger: Logger = req.app.state.logger
    history_messages = None
    if chat_request.history:
        history_messages = [[msg.question, msg.answer] for msg in chat_request.history]

    start = arrow.utcnow()

    response, history = req.app.state.llm.chat(
        req.app.state.tokenizer,
        chat_request.prompt,
        history=history_messages,
        max_length=chat_request.max_length if chat_request.max_length else 2048,
        top_p=chat_request.top_p if chat_request.top_p else 0.7,
        temperature=chat_request.temperature if chat_request.temperature else 0.95)
    # print(json.dumps(chat_request.dict(), indent=4))

    now = arrow.utcnow()
    history_messages = []
    if history and chat_request.with_history:
        history_messages = [QA(question=qa[0], answer=qa[1])for qa in history]

    resp = ChatGLMResponse(
        response=response,
        history=history_messages,
        start=start.isoformat(),
        end=now.isoformat(),
        took=(now-start).total_seconds()
    )

    logger.debug(f"Prompt: {chat_request.prompt} Response: {response}")
    # torch_gc()
    return resp


