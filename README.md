# Agentic AI RAG Chatbot — LangGraph + Pinecone

A custom Python Retrieval-Augmented Generation (RAG) chatbot built for the AI Engineer interview assignment. It uses the provided **Agentic AI eBook** as the only knowledge source.

## What is included

- PDF ingestion with page-aware metadata
- Recursive text chunking (default 800 characters, 100 overlap)
- OpenAI embeddings
- Pinecone vector storage
- LangGraph orchestration: **Retrieve → Generate → Grade → Finalize**
- Strict context-only generation
- Groundedness/confidence grading
- FastAPI JSON endpoint
- Automated unit tests
- Sample validation queries

The assignment explicitly requires ingestion/chunking, vector storage, LangGraph retrieval/generation/groundedness, and a structured API response. The implementation follows those requirements.

## Architecture

```text
Ebook-Agentic-AI.pdf
        |
        v
   PDF Parser
        |
        v
  Text Chunker
        |
        v
OpenAI Embeddings
        |
        v
     Pinecone
        |
        | query
        v
+-------------------+
| LangGraph         |
| Retrieve          |
|    ↓              |
| Generate          |
|    ↓              |
| Grounding Grade   |
|    ↓              |
| Finalize          |
+-------------------+
        |
        v
     FastAPI
        |
        v
Structured JSON
```

## Requirements

- Python 3.10+
- OpenAI API key
- Pinecone API key

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Add your real keys to `.env`.

### 4. Ingest the eBook

```bash
python -m scripts.ingest
```

The script parses the supplied 60-page eBook, creates page-aware chunks, embeds them, and upserts them into Pinecone.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 6. Query

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is the core definition of Agentic AI as outlined in the eBook?"}'
```

Response shape:

```json
{
  "query": "What is the core definition of Agentic AI as outlined in the eBook?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "..."
  ],
  "confidence_score": 0.91
}
```

## Groundedness behavior

The generation prompt explicitly forbids outside knowledge. The grading node checks whether the generated answer is supported by the retrieved context.

For an out-of-scope question such as:

> What is the capital of France?

the system is designed to refuse rather than answer from general model knowledge.

If retrieval or grounding confidence is insufficient, the finalization node returns a refusal.

## Validation queries

1. What is the core definition of Agentic AI as outlined in the eBook?
2. What are the main architectural components required to build agentic systems?
3. What real-world industry use cases for Agentic AI are discussed in the eBook?
4. How does Agentic AI differ from traditional generative AI chatbots according to the text?
5. What key challenges or limitations of Agentic AI are mentioned in the document?
6. What is the capital of France?

## Tests

Run:

```bash
pytest -q
```

Tests cover:

- PDF parsing
- 60-page source integrity
- chunk generation and metadata
- request/response schema
- graph construction without making API calls

## Project structure

```text
.
├── app/
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── main.py
│   ├── schemas.py
│   └── vector_store.py
├── data/
│   └── Ebook-Agentic-AI.pdf
├── scripts/
│   └── ingest.py
├── tests/
│   ├── test_api_schema.py
│   ├── test_graph_logic.py
│   └── test_ingestion.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Notes

- Pinecone is used as the vector database as requested.
- The embedding model `text-embedding-3-small` produces 1536-dimensional vectors, matching the configured Pinecone index.
- The code does not contain API keys or secrets.
- The source PDF is included under `data/` for reproducible ingestion.
