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

        print(
            f"Loading local embedding model: {embedding_model}"
        )

        self.embedder = SentenceTransformer(
            embedding_model
        )

        self.dimension = (
            self.embedder.get_sentence_embedding_dimension()
        )

        print(
            f"Embedding dimension: {self.dimension}"
        )

        existing = set(
            self.pc.list_indexes().names()
        )

        if index_name not in existing:
            print(
                f"Creating Pinecone index: {index_name}"
            )

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

    # ---------------------------------------------------------
    # EMBEDDINGS
    # ---------------------------------------------------------

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.embedder.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return vectors.tolist()

    # ---------------------------------------------------------
    # UPSERT
    # ---------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[dict],
        batch_size: int = 100,
    ) -> int:
        total = 0

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            batch = chunks[
                start:start + batch_size
            ]

            print(
                f"Embedding chunks "
                f"{start + 1}-"
                f"{start + len(batch)} "
                f"of {len(chunks)}..."
            )

            vectors = self.embed(
                [
                    c["text"]
                    for c in batch
                ]
            )

            payload = []

            for c, vector in zip(
                batch,
                vectors,
            ):
                payload.append(
                    {
                        "id": c["id"],
                        "values": vector,
                        "metadata": {
                            "text": c["text"],
                            "page_number": int(
                                c["page_number"]
                            ),
                            "source": (
                                "Ebook-Agentic-AI.pdf"
                            ),
                        },
                    }
                )

            self.index.upsert(
                vectors=payload,
                namespace=self.namespace,
            )

            total += len(payload)

        return total

    # ---------------------------------------------------------
    # QUERY EXPANSION
    # ---------------------------------------------------------

    def _expand_query(
        self,
        query: str,
    ) -> str:
        query_lower = query.lower()

        expanded = query

        # -----------------------------------------------------
        # CHALLENGES / LIMITATIONS
        # -----------------------------------------------------

        if (
            "challenge" in query_lower
            or "limitation" in query_lower
            or "problem" in query_lower
            or "difficulty" in query_lower
        ):
            expanded += (
                " "
                "Challenges and Mitigation Strategies "
                "Complex System Design "
                "Interoperability "
                "Data Security "
                "Conflict Resolution "
                "Scalability "
                "Cost of Implementation "
                "Slow Development "
                "Sophisticated Orchestration "
                "Communication and Coordination "
                "Conflict Management "
                "Choosing the Right Agent "
                "Reliability and Fault Tolerance "
                "multi-agent systems "
                "orchestrating complex agentic systems"
            )

        # -----------------------------------------------------
        # USE CASES / APPLICATIONS
        # -----------------------------------------------------

        if (
            "use case" in query_lower
            or "use cases" in query_lower
            or "application" in query_lower
            or "applications" in query_lower
            or "real-world" in query_lower
            or "real world" in query_lower
        ):
            expanded += (
                " "
                "Practical Applications of Agentic AI "
                "Industry Vertical Agents "
                "Healthcare "
                "Finance "
                "Retail "
                "Education "
                "Manufacturing "
                "Legal "
                "Transportation "
                "Biopharma "
                "Medical Devices "
                "Construction "
                "Drug Discovery "
                "Clinical Trial Optimization"
            )

        # -----------------------------------------------------
        # MULTI-AGENT SYSTEMS
        # -----------------------------------------------------

        if (
            "multi-agent" in query_lower
            or "multi agent" in query_lower
            or "mas" in query_lower
        ):
            expanded += (
                " "
                "Multi-Agent Systems "
                "Benefits "
                "Challenges "
                "Mitigation Strategies "
                "communication "
                "coordination "
                "task allocation "
                "orchestration "
                "conflict resolution "
                "scalability "
                "reliability "
                "fault tolerance"
            )

        # -----------------------------------------------------
        # ORCHESTRATION
        # -----------------------------------------------------

        if (
            "orchestrat" in query_lower
            or "coordination" in query_lower
            or "coordinate" in query_lower
        ):
            expanded += (
                " "
                "Orchestrating Agentic AI Systems "
                "Communication and Coordination "
                "Conflict Management "
                "Choosing the Right Agent "
                "Scalability "
                "Reliability and Fault Tolerance "
                "Agent Registries "
                "Task Breakdown "
                "Planning "
                "Verification"
            )

        # -----------------------------------------------------
        # AGENTIC AI DEFINITION
        # -----------------------------------------------------

        if (
            "what is agentic ai" in query_lower
            or "define agentic ai" in query_lower
            or "definition of agentic ai" in query_lower
        ):
            expanded += (
                " "
                "autonomous decision-making "
                "autonomous action "
                "specific objectives "
                "Perception "
                "Reasoning "
                "Planning "
                "Learning "
                "Execution"
            )

        return expanded

    # ---------------------------------------------------------
    # TOKEN EXTRACTION
    # ---------------------------------------------------------

    def _get_terms(
        self,
        text: str,
    ) -> set[str]:
        return set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                text.lower(),
            )
        )

    # ---------------------------------------------------------
    # QUERY
    # ---------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,
    ) -> list[RetrievedChunk]:

        # -----------------------------------------------------
        # EXPAND USER QUERY
        # -----------------------------------------------------

        retrieval_query = self._expand_query(
            query
        )

        # -----------------------------------------------------
        # CREATE QUERY EMBEDDING
        # -----------------------------------------------------

        query_vector = self.embed(
            [retrieval_query]
        )[0]

        # Retrieve many candidates first.
        #
        # We intentionally retrieve more than top_k because
        # the relevant detailed section may not be among the
        # first few semantic matches.
        candidate_k = max(
            top_k * 8,
            40,
        )

        result = self.index.query(
            vector=query_vector,
            top_k=candidate_k,
            include_metadata=True,
            namespace=self.namespace,
        )

        # -----------------------------------------------------
        # QUERY TERMS
        # -----------------------------------------------------

        query_terms = self._get_terms(
            query
        )

        expanded_terms = self._get_terms(
            retrieval_query
        )

        # -----------------------------------------------------
        # SPECIAL IMPORTANT TERMS
        # -----------------------------------------------------

        important_terms = {
            "challenge",
            "challenges",
            "limitation",
            "limitations",
            "interoperability",
            "security",
            "scalability",
            "conflict",
            "resolution",
            "communication",
            "coordination",
            "reliability",
            "fault",
            "tolerance",
            "orchestration",
            "orchestrating",
            "implementation",
            "development",
            "healthcare",
            "finance",
            "retail",
            "education",
            "manufacturing",
            "transportation",
            "biopharma",
            "construction",
            "legal",
            "applications",
            "application",
        }

        candidates = []

        for match in result.matches:

            semantic_score = float(
                match.score or 0.0
            )

            metadata = match.metadata or {}

            text = str(
                metadata.get(
                    "text",
                    "",
                )
            )

            if (
                not text
                or semantic_score < min_score
            ):
                continue

            text_lower = text.lower()

            # -------------------------------------------------
            # NORMAL LEXICAL MATCH
            # -------------------------------------------------

            lexical_hits = sum(
                1
                for term in expanded_terms
                if term in text_lower
            )

            lexical_score = min(
                lexical_hits
                / max(
                    len(expanded_terms),
                    1,
                ),
                1.0,
            )

            # -------------------------------------------------
            # IMPORTANT TERM MATCH
            # -------------------------------------------------

            important_hits = sum(
                1
                for term in important_terms
                if term in text_lower
            )

            important_score = min(
                important_hits / 8.0,
                1.0,
            )

            # -------------------------------------------------
            # DIRECT QUERY TERM MATCH
            # -------------------------------------------------

            direct_hits = sum(
                1
                for term in query_terms
                if term in text_lower
            )

            direct_score = min(
                direct_hits
                / max(
                    len(query_terms),
                    1,
                ),
                1.0,
            )

            # -------------------------------------------------
            # COMBINED SCORE
            # -------------------------------------------------

            combined_score = (
                0.55 * semantic_score
                + 0.20 * lexical_score
                + 0.15 * important_score
                + 0.10 * direct_score
            )

            candidates.append(
                (
                    combined_score,
                    RetrievedChunk(
                        text=text,
                        page_number=int(
                            metadata.get(
                                "page_number",
                                0,
                            )
                        ),
                        score=semantic_score,
                        chunk_id=str(
                            match.id
                        ),
                    ),
                )
            )

        # -----------------------------------------------------
        # SORT BY FINAL SCORE
        # -----------------------------------------------------

        candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # -----------------------------------------------------
        # DIVERSITY / DUPLICATE CONTROL
        # -----------------------------------------------------

        selected = []

        seen_text = set()

        for _, chunk in candidates:

            normalized_text = (
                chunk.text
                .strip()
                .lower()
            )

            # Avoid returning exactly duplicated chunks.
            if normalized_text in seen_text:
                continue

            seen_text.add(
                normalized_text
            )

            selected.append(chunk)

            if len(selected) >= top_k:
                break

        return selected
