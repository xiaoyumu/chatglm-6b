from typing import Optional, List

from pydantic import BaseModel


class QA(BaseModel):
    question: str
    answer: str


class ChatGLMChatRequest(BaseModel):
    prompt: str
    history: Optional[List[QA]]
    max_length: Optional[int] = 2048
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95

