"""
ingestion.py
------------
Handles reading PDF, DOCX, TXT files and splitting them into chunks.
Chunks are small pieces of text that will be embedded and stored.
"""

import fitz  # PyMuPDF for PDF parsing
import docx  # python-docx for Word files
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text(file_path: str) -> str:
    """
    Extract raw text from a file based on its extension.
    Supports: .pdf, .docx, .txt
    """
    ext = file_path.lower().split(".")[-1]

    if ext == "pdf":
        return _extract_pdf(file_path)
    elif ext == "docx":
        return _extract_docx(file_path)
    elif ext == "txt":
        return _extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def _extract_pdf(file_path: str) -> str:
    """Extract text from all pages of a PDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


def _extract_docx(file_path: str) -> str:
    """Extract text from all paragraphs of a DOCX file."""
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])


def _extract_txt(file_path: str) -> str:
    """Read plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()



def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """
    Split text into overlapping chunks.
    
    - chunk_size: max tokens per chunk (500 is a good balance)
    - chunk_overlap: how much text repeats between chunks (helps preserve context)
    
    We use RecursiveCharacterTextSplitter which tries to split on:
    paragraphs → sentences → words → characters (in that order)
    This keeps semantically related content together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Filter out very short/empty chunks
    return [c.strip() for c in chunks if len(c.strip()) > 50]


def ingest_document(file_path: str) -> list[str]:
    """
    Full ingestion pipeline:
    file → extract text → chunk → return chunks
    """
    print(f"[Ingestion] Reading: {file_path}")
    text = extract_text(file_path)
    print(f"[Ingestion] Extracted {len(text)} characters")

    chunks = chunk_text(text)
    print(f"[Ingestion] Created {len(chunks)} chunks")
    return chunks
