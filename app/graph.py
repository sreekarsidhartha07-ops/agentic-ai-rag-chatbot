import json
import re
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from .schemas import QueryResponse
from .vector_store import PineconeStore, RetrievedChunk


class RAGState(TypedDict, total=False):
    query: str
    retrieved: list[RetrievedChunk]
    final_answer: str
    confidence_score: float
    grounded: bool


SYSTEM_PROMPT = """You are a strictly grounded RAG assistant.

You MUST answer ONLY from the supplied Agentic AI eBook context.

Do not use outside knowledge.

If the context does not contain enough information to answer the question,
say exactly:

"I’m sorry, but that information is not available in the provided eBook."

Keep the answer concise and factual.
"""


GRADE_PROMPT = """You are a strict groundedness grader.

Check whether the proposed answer is supported by the supplied Agentic AI
eBook context.

Return ONLY valid JSON:

{{
  "grounded": true,
  "confidence": 0.95
}}

Rules:

- grounded must be true or false.
- confidence must be between 0.0 and 1.0.
- A refusal because the context lacks the answer is grounded=true.
- Do not use outside knowledge.
"""


def _context_text(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[Page {c.page_number} | {c.chunk_id}]\n{c.text}"
        for c in chunks
    )


def _parse_grade(raw: str) -> tuple[bool, float]:
    raw = raw.strip()

    # Remove Markdown code fences if Ollama returns JSON inside ```json ... ```
    raw = re.sub(
        r"```json\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(
        r"```\s*",
        "",
        raw,
    )

    # First try to parse the complete response as JSON.
    try:
        data = json.loads(raw)

        confidence = max(
            0.0,
            min(
                1.0,
                float(data.get("confidence", 0.0)),
            ),
        )

        grounded = bool(
            data.get("grounded", False)
        )

        return grounded, confidence

    except (
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ):
        pass

    # If there is extra text around the JSON, extract the JSON object.
    match = re.search(
        r"\{.*\}",
        raw,
        flags=re.DOTALL,
    )

    if match:
        try:
            data = json.loads(match.group(0))

            confidence = max(
                0.0,
                min(
                    1.0,
                    float(data.get("confidence", 0.0)),
                ),
            )

            grounded = bool(
                data.get("grounded", False)
            )

            return grounded, confidence

        except (
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            pass

    return False, 0.0


def build_graph(
    store: PineconeStore,
    chat_model: str,
    grounding_threshold: float = 0.70,
    top_k: int = 5,
    min_retrieval_score: float = 0.35,
    ollama_base_url: str = "http://localhost:11434",
    llm=None,
):
    # Use local Ollama instead of OpenAI.
    llm = llm or ChatOllama(
        model=chat_model,
        temperature=0,
        base_url=ollama_base_url,
    )

    # ---------------------------------------------------------
    # RETRIEVE
    # ---------------------------------------------------------

    def retrieve(state: RAGState):
        chunks = store.query(
            state["query"],
            top_k=top_k,
            min_score=min_retrieval_score,
        )

        return {
            "retrieved": chunks,
        }

    # ---------------------------------------------------------
    # GENERATE
    # ---------------------------------------------------------

    def generate(state: RAGState):
        chunks = state.get(
            "retrieved",
            [],
        )

        # No relevant chunks found.
        if not chunks:
            return {
                "final_answer": (
                    "I’m sorry, but that information is "
                    "not available in the provided eBook."
                ),
                "confidence_score": 0.0,
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    SYSTEM_PROMPT,
                ),
                (
                    "human",
                    """Question:

{query}

Context:

{context}

Answer only from the supplied context.""",
                ),
            ]
        )

        response = (
            prompt | llm
        ).invoke(
            {
                "query": state["query"],
                "context": _context_text(chunks),
            }
        )

        return {
            "final_answer": response.content.strip(),
        }

    # ---------------------------------------------------------
    # GRADE / VERIFY
    # ---------------------------------------------------------

    def grade(state: RAGState):
        chunks = state.get(
            "retrieved",
            [],
        )

        if not chunks:
            return {
                "grounded": True,
                "confidence_score": 0.0,
            }

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    GRADE_PROMPT,
                ),
                (
                    "human",
                    """Question:

{query}

Context:

{context}

Answer:

{answer}""",
                ),
            ]
        )

        response = (
            prompt | llm
        ).invoke(
            {
                "query": state["query"],
                "context": _context_text(chunks),
                "answer": state["final_answer"],
            }
        )

        grounded, confidence = _parse_grade(
            response.content
        )

        return {
            "grounded": grounded,
            "confidence_score": confidence,
        }

    # ---------------------------------------------------------
    # FINALIZE
    # ---------------------------------------------------------

    def finalize(state: RAGState):
        grounded = state.get(
            "grounded",
            False,
        )

        confidence = state.get(
            "confidence_score",
            0.0,
        )

        # Reject answers that cannot be verified.
        if (
            not grounded
            or confidence < grounding_threshold
        ):
            return {
                "final_answer": (
                    "I’m sorry, but I can’t verify "
                    "that answer from the provided eBook."
                )
            }

        return {}

    # ---------------------------------------------------------
    # LANGGRAPH WORKFLOW
    # ---------------------------------------------------------

    workflow = StateGraph(RAGState)

    workflow.add_node(
        "retrieve",
        retrieve,
    )

    workflow.add_node(
        "generate",
        generate,
    )

    workflow.add_node(
        "grade",
        grade,
    )

    workflow.add_node(
        "finalize",
        finalize,
    )

    workflow.set_entry_point(
        "retrieve"
    )

    workflow.add_edge(
        "retrieve",
        "generate",
    )

    workflow.add_edge(
        "generate",
        "grade",
    )

    workflow.add_edge(
        "grade",
        "finalize",
    )

    workflow.add_edge(
        "finalize",
        END,
    )

    return workflow.compile()


class RAGService:
    def __init__(
        self,
        store: PineconeStore,
        chat_model: str,
        grounding_threshold: float = 0.70,
        top_k: int = 5,
        min_retrieval_score: float = 0.35,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.graph = build_graph(
            store=store,
            chat_model=chat_model,
            grounding_threshold=grounding_threshold,
            top_k=top_k,
            min_retrieval_score=min_retrieval_score,
            ollama_base_url=ollama_base_url,
        )

    def ask(
        self,
        query: str,
    ) -> QueryResponse:
        state = self.graph.invoke(
            {
                "query": query.strip(),
            }
        )

        chunks = state.get(
            "retrieved",
            [],
        )

        return QueryResponse(
            query=query.strip(),
            final_answer=state.get(
                "final_answer",
                "",
            ),
            retrieved_context_chunks=[
                c.text for c in chunks
            ],
            confidence_score=round(
                float(
                    state.get(
                        "confidence_score",
                        0.0,
                    )
                ),
                4,
            ),
        )
