from typing import Optional, List

from pydantic import BaseModel


class QA(BaseModel):
    question: str
    answer: str


class ChatGLMRequest(BaseModel):
    prompt: str
    history: Optional[List[QA]]
    max_length: Optional[int] = 2048
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95
    with_history: Optional[bool] = True


class ChatGLMResponse(BaseModel):
    response: Optional[str] = None
    history: Optional[List[QA]] = None
    start: Optional[str] = None
    end: Optional[str] = None
    took: Optional[float] = None  # Seconds
