import argparse
from pathlib import Path

from app.config import get_settings
from app.ingestion import build_chunks
from app.vector_store import PineconeStore


def main():
    parser = argparse.ArgumentParser(
        description="Ingest the Agentic AI eBook into Pinecone."
    )

    parser.add_argument(
        "--pdf",
        default="data/Ebook-Agentic-AI.pdf",
    )

    args = parser.parse_args()

    settings = get_settings()

    print("Reading PDF...")

    chunks = build_chunks(
        args.pdf,
        settings.chunk_size,
        settings.chunk_overlap,
    )

    print(f"Created {len(chunks)} chunks.")

    store = PineconeStore(
        pinecone_api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        embedding_model=settings.embedding_model,
    )

    count = store.upsert_chunks(chunks)

    print()
    print(
        f"Ingested {count} chunks from "
        f"{Path(args.pdf).name}."
    )

    print(
        f"Index: {settings.pinecone_index_name}"
    )

    print(
        f"Namespace: {settings.pinecone_namespace}"
    )


if __name__ == "__main__":
    main()
