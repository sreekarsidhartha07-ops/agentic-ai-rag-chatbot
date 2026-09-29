from app.graph import build_graph

class FakeStore:
    def query(self, query, top_k=5, min_score=0.35):
        from app.vector_store import RetrievedChunk
        return [RetrievedChunk(
            text="Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
            page_number=18, score=0.95, chunk_id="chunk-1"
        )]

class FakeLLM:
    def invoke(self, messages):
        class Response:
            content = '{"grounded": true, "confidence": 0.9}'
        return Response()

def test_graph_can_be_constructed_without_network_calls():
    graph = build_graph(FakeStore(), "test-model", llm=FakeLLM())
    assert graph is not None
