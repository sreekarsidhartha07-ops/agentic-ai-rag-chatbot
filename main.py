from fastapi import FastAPI, HTTPException

from .config import get_settings
from .schemas import QueryRequest, QueryResponse
from .vector_store import PineconeStore
from .graph import RAGService


settings = get_settings()

app = FastAPI(
    title=settings.app_title,
    version="1.0.0",
)

_service: RAGService | None = None


def get_service() -> RAGService:
    global _service

    if _service is None:
        store = PineconeStore(
            pinecone_api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            namespace=settings.pinecone_namespace,
            embedding_model=settings.embedding_model,
        )

        _service = RAGService(
            store=store,
            chat_model=settings.ollama_model,
            grounding_threshold=settings.grounding_threshold,
            top_k=settings.top_k,
            min_retrieval_score=settings.min_retrieval_score,
            ollama_base_url=settings.ollama_base_url,
        )

    return _service


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        return get_service().ask(request.query)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc