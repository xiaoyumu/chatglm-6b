import datetime
import json

from fastapi import APIRouter
from fastapi import Request

from api.model.llm import ChatGLMChatRequest

router = APIRouter(prefix="/api/llm", tags=["LLM"])


@router.post("/chat", summary="Chat with ChatGLM model.")
async def send_chat_request(req: Request, chat_request: ChatGLMChatRequest):
    history_messages = None
    if chat_request.history:
        history_messages = [(msg.question, msg.answer) for msg in chat_request.history]
    response, history = req.app.state.llm.chat(
        req.app.state.tokenizer,
        chat_request.prompt,
        history=history_messages,
        max_length=chat_request.max_length if chat_request.max_length else 2048,
        top_p=chat_request.top_p if chat_request.top_p else 0.7,
        temperature=chat_request.temperature if chat_request.temperature else 0.95)
    print(json.dumps(chat_request.dict(), indent=4))
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + chat_request.prompt + '", response:"' + repr(response) + '"'
    print(log)
    # torch_gc()
    return answer


