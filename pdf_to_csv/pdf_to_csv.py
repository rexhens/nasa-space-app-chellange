
#!/usr/bin/env python3
"""
NASA Space App Challenge - PDF to CSV Converter
===============================================

Converts a folder of PDF research papers into structured CSV datasets with columns:
- paper_id, title, abstract, results, conclusion, full_text
- Also creates a CHUNKS CSV with paragraph-sized chunks for embeddings

Usage:
    pip install pymupdf pandas tqdm
    python pdf_to_csv.py --pdf_dir ./papers/pdf_papers --out_csv ./papers/csv_papers/spacebio_papers.csv --chunks_csv ./papers/csv_chunks/spacebio_chunks.csv

Author: NASA Space App Challenge Team
"""

import os
import re
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Section detection regex pattern
SECTION_RE = re.compile(r"^\s*(abstract|introduction|background|methods?|materials|results?|discussion|conclusion[s]?|references)\s*[:\.]?\s*$", re.I)

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text content from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text content or empty string if extraction fails
    """
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page_num, page in enumerate(doc):
            try:
                txt = page.get_text("text")
                if txt and txt.strip():
                    texts.append(txt)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num} in {pdf_path}: {e}")
                continue
        doc.close()
        return "\n".join(texts)
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        return ""

def guess_title(first_page_text: str) -> str:
    """
    Attempt to extract the paper title from the first page text.
    
    Args:
        first_page_text: Text content from the first page
        
    Returns:
        Guessed title or empty string if not found
    """
    lines = first_page_text.splitlines()
    
    # Look for title in first 12 lines (most papers have title early)
    for line in lines[:12]:
        line = line.strip()
        if len(line) > 5 and not re.fullmatch(r"[A-Z\s\-–:,\\.0-9]+", line):
            return line
    
    # Fallback: return first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    
    return ""

def split_into_sections(raw_text: str) -> Tuple[str, str, str, str, str]:
    """
    Split PDF text into structured sections: title, abstract, results, conclusion, full_text.
    
    Args:
        raw_text: Raw text extracted from PDF
        
    Returns:
        Tuple of (title, abstract, results, conclusion, full_text)
    """
    # Clean up text
    text = re.sub(r"\r", "", raw_text)
    lines = text.splitlines()
    sections = []
    current_name = "preamble"
    current_buf = []

    def push_section(name: str, buf: List[str]) -> None:
        """Add a section to the sections list if it has content."""
        if buf:
            sections.append((name.lower(), "\n".join(buf).strip()))

    # Parse sections based on headers
    for ln in lines:
        if SECTION_RE.match(ln.strip()):
            push_section(current_name, current_buf)
            current_name = SECTION_RE.match(ln.strip()).group(1).lower()
            current_buf = []
        else:
            current_buf.append(ln)
    push_section(current_name, current_buf)

    def pick(name: str) -> str:
        """Extract content from a specific section."""
        for sec_name, sec_text in sections:
            if sec_name == name:
                return sec_text
        return ""

    # Extract title from first page
    first_page = "\n".join(lines[:150])
    title = guess_title(first_page)

    # Extract main sections
    abstract = pick("abstract")
    results = pick("results")
    conclusion = pick("conclusion") or pick("conclusions")

    # Fallback regex patterns for sections not caught by header detection
    if not abstract:
        m = re.search(r"(?:^|\n)\s*abstract\s*[:\.]?\s*(.+?)(?:\n\s*(?:introduction|background|methods?|materials)\b|$)", text, re.I | re.S)
        if m:
            abstract = m.group(1).strip()

    if not results:
        m = re.search(r"(?:^|\n)\s*results?\s*[:\.]?\s*(.+?)(?:\n\s*(?:discussion|conclusion[s]?|references)\b|$)", text, re.I | re.S)
        if m:
            results = m.group(1).strip()

    if not conclusion:
        m = re.search(r"(?:^|\n)\s*conclusion[s]?\s*[:\.]?\s*(.+?)(?:\n\s*(?:references|acknowledgments?|supplementary)\b|$)", text, re.I | re.S)
        if m:
            conclusion = m.group(1).strip()

    return title, abstract, results, conclusion, text.strip()

def paragraph_chunks(text: str, max_chars: int = 900, min_chars: int = 250) -> List[str]:
    """
    Split text into paragraph-sized chunks for embedding generation.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        min_chars: Minimum characters per chunk
        
    Yields:
        Text chunks suitable for embedding
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunk = ""
    
    for p in paras:
        if len(chunk) + len(p) + 2 <= max_chars:
            chunk = (chunk + "\n\n" + p).strip()
        else:
            if len(chunk) >= min_chars:
                yield chunk
                chunk = p
            else:
                chunk = (chunk + "\n\n" + p).strip()
                if len(chunk) >= min_chars:
                    yield chunk
                    chunk = ""
    
    if chunk:
        yield chunk

def build_datasets(pdf_dir: str, out_csv: str, chunks_csv: Optional[str] = None) -> None:
    """
    Process PDF files and create structured CSV datasets.
    
    Args:
        pdf_dir: Directory containing PDF files
        out_csv: Output path for main papers CSV
        chunks_csv: Output path for chunks CSV (optional)
    """
    records = []
    chunk_records = []
    pdf_paths = sorted([p for p in Path(pdf_dir).glob("**/*.pdf")])
    
    logger.info(f"Found {len(pdf_paths)} PDF files to process")

    for p in tqdm(pdf_paths, desc="Processing PDFs"):
        try:
            raw = extract_text_from_pdf(p)
            if not raw:
                logger.warning(f"No text extracted from {p}")
                continue
                
            title, abstract, results, conclusion, full_text = split_into_sections(raw)
            paper_id = p.stem

            # Create main record
            records.append({
                "paper_id": paper_id,
                "filename": str(p),
                "title": title,
                "abstract": abstract,
                "results": results,
                "conclusion": conclusion,
                "full_text": full_text
            })

            # Create chunks for embedding
            if chunks_csv:
                # Use structured content for chunks (abstract + results + conclusion)
                structured_content = "\n\n".join([s for s in [abstract, results, conclusion] if s]).strip()
                base_for_chunks = structured_content if structured_content else full_text
                
                for i, chunk in enumerate(paragraph_chunks(base_for_chunks)):
                    chunk_records.append({
                        "paper_id": paper_id,
                        "chunk_id": f"{paper_id}_{i:03d}",
                        "text": chunk
                    })
                    
        except Exception as e:
            logger.error(f"Error processing {p}: {e}")
            continue

    # Create output directories if they don't exist
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if chunks_csv:
        chunks_csv_path = Path(chunks_csv)
        chunks_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Save datasets
    logger.info(f"Saving {len(records)} papers to {out_csv}")
    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    
    if chunks_csv and len(chunk_records) > 0:
        logger.info(f"Saving {len(chunk_records)} chunks to {chunks_csv}")
        pd.DataFrame.from_records(chunk_records).to_csv(chunks_csv, index=False)
    elif chunks_csv:
        logger.warning("No chunks generated - check if PDFs contain extractable text")

def main():
    """Main entry point for the PDF to CSV converter."""
    parser = argparse.ArgumentParser(
        description="Convert PDF research papers to structured CSV datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_to_csv.py --pdf_dir ./papers/pdf_papers
  python pdf_to_csv.py --pdf_dir ./papers/pdf_papers --out_csv ./data/papers.csv --chunks_csv ./data/chunks.csv
        """
    )
    
    parser.add_argument(
        "--pdf_dir", 
        type=str, 
        required=True, 
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--out_csv", 
        type=str, 
        default="./papers/csv_papers/spacebio_papers.csv",
        help="Output path for main papers CSV file"
    )
    parser.add_argument(
        "--chunks_csv", 
        type=str, 
        default="./papers/csv_chunks/spacebio_chunks.csv",
        help="Output path for chunks CSV file (optional)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return 1
    
    if not pdf_dir.is_dir():
        logger.error(f"Path is not a directory: {pdf_dir}")
        return 1
    
    try:
        build_datasets(args.pdf_dir, args.out_csv, args.chunks_csv)
        logger.info("PDF processing completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
