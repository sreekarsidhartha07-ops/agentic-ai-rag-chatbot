# Agentic AI RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **LangGraph, Pinecone, Sentence Transformers, Ollama, and FastAPI**.

The chatbot answers questions strictly from the provided **Agentic AI eBook** and refuses questions when sufficient information cannot be found in the eBook.

## Features

- PDF document ingestion
- Text extraction from the Agentic AI eBook
- Recursive text chunking
- Local embeddings using Sentence Transformers
- Vector storage and retrieval using Pinecone
- Semantic + lexical retrieval reranking
- LangGraph-based RAG workflow
- Local LLM using Ollama
- Groundedness and confidence verification
- Out-of-scope question refusal
- FastAPI REST API
- Automated tests

## Architecture

```text
Agentic AI eBook PDF
        |
        v
   PDF Extraction
        |
        v
   Text Chunking
        |
        v
Sentence Transformers
        |
        v
     Embeddings
        |
        v
     Pinecone
        |
        v
     Retrieval
        |
        v
    LangGraph
        |
        v
     Generate
        |
        v
   Verify / Grade
        |
        v
   Final Answer
```

## LangGraph Workflow

```text
START
  |
  v
Retrieve
  |
  v
Generate
  |
  v
Grade / Verify
  |
  v
Finalize
  |
  v
END
```

The verification step checks whether the generated answer is sufficiently grounded in the retrieved eBook context.

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| RAG Orchestration | LangGraph |
| Vector Database | Pinecone |
| Embeddings | Sentence Transformers |
| Local LLM | Ollama / Llama 3.2 3B |
| API | FastAPI |
| PDF Processing | pypdf |
| Text Chunking | LangChain Text Splitters |
| Testing | Pytest |

## Project Structure

```text
agentic-ai-rag-chatbot/
|
├── app/
│   ├── config.py
│   ├── graph.py
│   ├── ingestion.py
│   ├── main.py
│   ├── schemas.py
│   └── vector_store.py
|
├── data/
│   └── Ebook-Agentic-AI.pdf
|
├── scripts/
│   └── ingest.py
|
├── tests/
│   ├── test_api.py
│   ├── test_graph.py
│   └── test_ingestion.py
|
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sreekarsidhartha07-ops/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file using `.env.example`.

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-rag-free
PINECONE_NAMESPACE=agentic-ai
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
OLLAMA_MODEL=llama3.2:3b
OLLAMA_BASE_URL=http://localhost:11434
```

**Never commit the real `.env` file or API keys to GitHub.**

## Ollama Setup

Install Ollama and download the required model:

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before starting the API.

## Ingest the eBook

```bash
python -m scripts.ingest
```

The ingestion pipeline loads the PDF, extracts text, splits it into chunks, generates local embeddings, and stores vectors and metadata in Pinecone.

## Run the API

```bash
uvicorn app.main:app --reload
```

API: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

## API

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

### Query

```http
POST /query
```

Example request:

```json
{
  "query": "What is Agentic AI?"
}
```

Example response:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to systems capable of autonomous decision-making and action in pursuit of specific objectives.",
  "retrieved_context_chunks": [
    "Relevant content retrieved from the Agentic AI eBook..."
  ],
  "confidence_score": 0.95
}
```

## RAG Pipeline

```text
User Query
    |
    v
Query Embedding
    |
    v
Pinecone Retrieval
    |
    v
Context Reranking
    |
    v
LangGraph
    |
    v
LLM Generation
    |
    v
Groundedness Verification
    |
    v
Final Response
```

## Grounding and Out-of-Scope Handling

The chatbot is designed to answer questions only from the provided Agentic AI eBook.

If the retrieved context does not provide sufficient information, the chatbot refuses to answer instead of relying on outside knowledge.

For example:

```text
What is the capital of France?
```

This question is outside the scope of the Agentic AI eBook and should be refused.

## Example Questions

- What is Agentic AI?
- What are the key components of Agentic AI?
- How is Agentic AI different from traditional AI?
- What are the challenges of multi-agent systems?
- What are the challenges of orchestrating complex agentic systems?
- What are the practical applications of Agentic AI?
- What are the benefits of multi-agent systems?
- What are industry vertical agents?

## Testing

```bash
pytest -q
```

The tests cover PDF ingestion, API response structure, LangGraph logic, and RAG workflow behavior.

## Security

Sensitive environment variables are excluded from version control using `.gitignore`.

Never commit:

```text
.env
```

The repository only contains `.env.example` with placeholder values.

## Why Local AI?

This project uses:

- **Sentence Transformers** for local embeddings
- **Ollama / Llama 3.2 3B** for local LLM inference
- **Pinecone** for vector storage
- **LangGraph** for workflow orchestration

This avoids dependency on paid OpenAI API credits for the LLM and embedding generation.

## Key Design Decisions

### Chunking

The eBook is split using recursive text splitting with:

```text
Chunk size: 800 characters
Chunk overlap: 100 characters
```

### Retrieval

The system retrieves multiple candidate chunks from Pinecone and applies semantic and lexical relevance scoring before sending the most relevant context to the generation step.

### Verification

The LangGraph workflow evaluates:

- Whether the answer is grounded in retrieved context
- Whether the retrieved information is sufficient
- A confidence score for the final response

## Expected Response Format

```json
{
  "query": "...",
  "final_answer": "...",
  "retrieved_context_chunks": [
    "..."
  ],
  "confidence_score": 0.92
}
```

## Author

**Sreekar Sidhartha**

B.Tech Computer Science and Engineering — 2026
