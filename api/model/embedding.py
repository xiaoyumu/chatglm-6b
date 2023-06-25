from typing import Optional, List
from pydantic import BaseModel


class TextEmbeddingRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    