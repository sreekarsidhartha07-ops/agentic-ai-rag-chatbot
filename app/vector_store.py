import re
from dataclasses import dataclass
from typing import Sequence

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    page_number: int
    score: float
    chunk_id: str


class PineconeStore:
    def __init__(
        self,
        *,
        pinecone_api_key: str,
        index_name: str,
        namespace: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cloud: str = "aws",
        region: str = "us-east-1",
    ):
        self.pc = Pinecone(api_key=pinecone_api_key)

        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model

        print(f"Loading local embedding model: {embedding_model}")

        self.embedder = SentenceTransformer(embedding_model)

        self.dimension = self.embedder.get_sentence_embedding_dimension()

        print(f"Embedding dimension: {self.dimension}")

        existing = set(self.pc.list_indexes().names())

        if index_name not in existing:
            print(f"Creating Pinecone index: {index_name}")

            self.pc.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region,
                ),
            )

        self.index = self.pc.Index(index_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.embedder.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    def upsert_chunks(
        self,
        chunks: Sequence[dict],
        batch_size: int = 100,
    ) -> int:
        total = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]

            print(
                f"Embedding chunks "
                f"{start + 1}-{start + len(batch)} "
                f"of {len(chunks)}..."
            )

            vectors = self.embed([c["text"] for c in batch])

            payload = []

            for c, vector in zip(batch, vectors):
                payload.append(
                    {
                        "id": c["id"],
                        "values": vector,
                        "metadata": {
                            "text": c["text"],
                            "page_number": int(c["page_number"]),
                            "source": "Ebook-Agentic-AI.pdf",
                        },
                    }
                )

            self.index.upsert(
                vectors=payload,
                namespace=self.namespace,
            )

            total += len(payload)

        return total

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[RetrievedChunk]:
        query_vector = self.embed([query])[0]

        candidate_k = max(top_k * 4, 20)

        result = self.index.query(
            vector=query_vector,
            top_k=candidate_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        query_terms = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                query.lower(),
            )
        )

        expansions = {
            "challenges": {
                "challenge",
                "challenges",
                "limitation",
                "limitations",
                "risk",
                "risks",
                "problem",
                "problems",
            },
            "challenge": {
                "challenge",
                "challenges",
                "limitation",
                "limitations",
                "risk",
                "risks",
            },
            "use": {
                "use",
                "uses",
                "usecase",
                "usecases",
                "applications",
                "application",
                "industry",
                "industries",
            },
            "cases": {
                "usecase",
                "usecases",
                "applications",
                "application",
            },
            "agentic": {
                "agentic",
                "agent",
                "agents",
                "multi-agent",
                "multiagent",
                "orchestration",
            },
        }

        expanded_terms = set(query_terms)

        for term in query_terms:
            expanded_terms.update(
                expansions.get(term, set())
            )

        candidates = []

        for match in result.matches:
            semantic_score = float(match.score or 0.0)
            metadata = match.metadata or {}
            text = str(metadata.get("text", ""))

            if semantic_score < min_score or not text:
                continue

            text_lower = text.lower()

            lexical_hits = sum(
                1
                for term in expanded_terms
                if term in text_lower
            )

            lexical_boost = min(
                lexical_hits / max(len(expanded_terms), 1),
                1.0,
            )

            combined_score = (
                0.75 * semantic_score
                + 0.25 * lexical_boost
            )

            candidates.append(
                (
                    combined_score,
                    RetrievedChunk(
                        text=text,
                        page_number=int(
                            metadata.get("page_number", 0)
                        ),
                        score=semantic_score,
                        chunk_id=str(match.id),
                    ),
                )
            )

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            chunk
            for _, chunk in candidates[:top_k]
        ]
