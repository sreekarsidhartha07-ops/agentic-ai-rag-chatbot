from pathlib import Path

from app.ingestion import build_chunks, load_pdf_pages, chunk_pages

PDF = Path("data/Ebook-Agentic-AI.pdf")

def test_pdf_has_text():
    pages = load_pdf_pages(PDF)
    assert len(pages) == 60
    assert any("Agentic AI" in p["text"] for p in pages)

def test_chunking_has_metadata_and_overlap():
    chunks = chunk_pages([{"page_number": 1, "text": "A " * 1000}], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c["id"].startswith("chunk-") for c in chunks)
    assert all(c["page_number"] == 1 for c in chunks)
    assert all(c["text"] for c in chunks)
