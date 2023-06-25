from typing import Optional, List

from fastapi import FastAPI, Request
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from pydantic import BaseModel
from torch import hub
from transformers import AutoTokenizer, AutoModel
import uvicorn, json, datetime
import torch

MODEL_HOME = ".\\THUDM\\chatglm-6b"

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


class QA(BaseModel):
    question: str
    answer: str


class ChatGLMChatRequest(BaseModel):
    prompt: str
    history: Optional[List[QA]]
    max_length: Optional[int] = 2048
    top_p: Optional[float] = 0.7
    temperature: Optional[float] = 0.95


class TextEmbeddingRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


app = FastAPI()


@app.post("/chat", summary="Chat with ChatGLM model.")
async def send_chat_request(request: Request, chat_request: ChatGLMChatRequest):
    global model, tokenizer
    history_messages = None
    if chat_request.history:
        history_messages = [(msg.question, msg.answer) for msg in chat_request.history]
    response, history = model.chat(tokenizer,
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
    torch_gc()
    return answer


@app.post("/embeddings", summary="Get Text embeddings.")
async def get_text_embeddings(request: Request, embedding_request: TextEmbeddingRequest):
    global embedding

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(time)

    if embedding_request.text:
        embeddings = embedding.embed_documents([embedding_request.text])
    elif embedding_request.texts:
        embeddings = embedding.embed_documents(embedding_request.texts)
    else:
        embeddings = []
    answer = {
        "embeddings": embeddings,
        "text": embedding_request.text,
        "status": 200,
        "time": time
    }
    torch_gc()
    return answer


if __name__ == '__main__':
    # Load torch hub model from ./torch
    # For embeddings
    hub.set_dir("./torch")
    if torch.cuda.is_available():
        print("GPU Found. Running in GPU mode.")
    else:
        print("GPU Not Found. Running in CPU mode.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_HOME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_HOME, trust_remote_code=True).half().cuda()
    model.eval()
    embedding = HuggingFaceInstructEmbeddings()
    uvicorn.run(app, host='localhost', port=8000, workers=1)
