from app.schemas import QueryRequest, QueryResponse

def test_request_validation():
    q = QueryRequest(query="What is Agentic AI?")
    assert q.query == "What is Agentic AI?"

def test_response_shape():
    r = QueryResponse(
        query="x",
        final_answer="y",
        retrieved_context_chunks=["z"],
        confidence_score=0.9,
    )
    assert r.confidence_score == 0.9
