from pathlib import Path
from typing import Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    reader = PdfReader(str(path))
    pages: list[dict] = []
    has_text = False
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            has_text = True
        # Keep every original page number so metadata remains faithful to the PDF.
        pages.append({"page_number": page_number, "text": text})
    if not has_text:
        raise ValueError("No extractable text found in the PDF.")
    return pages

def chunk_pages(
    pages: Iterable[dict],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
) -> list[dict]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    chunk_id = 0

    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for text in page_chunks:
            cleaned = " ".join(text.split())
            if not cleaned:
                continue
            chunks.append({
                "id": f"chunk-{chunk_id:05d}",
                "text": cleaned,
                "page_number": page["page_number"],
            })
            chunk_id += 1
    if not chunks:
        raise ValueError("No chunks were created from the PDF.")
    return chunks

def build_chunks(pdf_path: str | Path, chunk_size: int = 800, chunk_overlap: int = 100) -> list[dict]:
    return chunk_pages(load_pdf_pages(pdf_path), chunk_size, chunk_overlap)
