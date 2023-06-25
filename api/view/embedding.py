import datetime

from fastapi import APIRouter
from fastapi import Request
from langchain.embeddings import HuggingFaceInstructEmbeddings

from api.model.embedding import TextEmbeddingRequest

router = APIRouter(prefix="/api/embeddings", tags=["Embeddings"])


@router.post("", summary="Get Text embeddings.")
async def get_text_embeddings(req: Request, embedding_request: TextEmbeddingRequest):
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(time)
    embedding_ctrl: HuggingFaceInstructEmbeddings = req.app.state.embedding
    if embedding_request.text:
        embeddings = embedding_ctrl.embed_documents([embedding_request.text])
    elif embedding_request.texts:
        embeddings = embedding_ctrl.embed_documents(embedding_request.texts)
    else:
        embeddings = []
    answer = {
        "embeddings": embeddings,
        "text": embedding_request.text,
        "status": 200,
        "time": time
    }
    return answer
