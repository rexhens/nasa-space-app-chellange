
# pdf_to_spacebio_csv.py
# Convert a folder of PDFs into CSV/JSONL-style datasets with columns:
# paper_id, title, abstract, results, conclusion, full_text
# Also creates a CHUNKS CSV with paragraph-sized chunks for embeddings.
#
# Usage:
#   pip install pymupdf pandas regex tqdm
#   python pdf_to_spacebio_csv.py --pdf_dir ./papers/pdf_papers --out_csv ./papers/csv_papers/spacebio_papers.csv --chunks_csv ./papers/csv_chunks/spacebio_chunks.csv

import os, re, argparse
from pathlib import Path
import pandas as pd
import fitz  # PyMuPDF
from tqdm import tqdm

SECTION_RE = re.compile(r"^\\s*(abstract|introduction|background|methods?|materials|results?|discussion|conclusion[s]?|references)\\s*[:\\.]?\\s*$", re.I)

def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            txt = page.get_text("text")
            if txt:
                texts.append(txt)
        doc.close()
        return "\\n".join(texts)
    except Exception:
        return ""

def guess_title(first_page_text: str) -> str:
    for line in first_page_text.splitlines()[:12]:
        line = line.strip()
        if len(line) > 5 and not re.fullmatch(r"[A-Z\\s\\-–:,\\.0-9]+", line):
            return line
    for line in first_page_text.splitlines():
        if line.strip():
            return line.strip()
    return ""

def split_into_sections(raw_text: str):
    text = re.sub(r"\\r", "", raw_text)
    lines = text.splitlines()
    sections = []
    current_name = "preamble"
    current_buf = []

    def push_section(name, buf):
        if buf:
            sections.append((name.lower(), "\\n".join(buf).strip()))

    for ln in lines:
        if SECTION_RE.match(ln.strip()):
            push_section(current_name, current_buf)
            current_name = SECTION_RE.match(ln.strip()).group(1).lower()
            current_buf = []
        else:
            current_buf.append(ln)
    push_section(current_name, current_buf)

    def pick(name):
        for sec_name, sec_text in sections:
            if sec_name == name:
                return sec_text
        return ""

    first_page = "\\n".join(lines[:150])
    title = guess_title(first_page)

    abstract = pick("abstract")
    results = pick("results")
    conclusion = pick("conclusion") or pick("conclusions")

    if not abstract:
        m = re.search(r"(?:^|\\n)\\s*abstract\\s*[:\\.]?\\s*(.+?)(?:\\n\\s*(?:introduction|background|methods?|materials)\\b|$)", text, re.I | re.S)
        if m:
            abstract = m.group(1).strip()

    if not results:
        m = re.search(r"(?:^|\\n)\\s*results?\\s*[:\\.]?\\s*(.+?)(?:\\n\\s*(?:discussion|conclusion[s]?|references)\\b|$)", text, re.I | re.S)
        if m:
            results = m.group(1).strip()

    if not conclusion:
        m = re.search(r"(?:^|\\n)\\s*conclusion[s]?\\s*[:\\.]?\\s*(.+?)(?:\\n\\s*(?:references|acknowledgments?|supplementary)\\b|$)", text, re.I | re.S)
        if m:
            conclusion = m.group(1).strip()

    return title, abstract, results, conclusion, text.strip()

def paragraph_chunks(text: str, max_chars=900, min_chars=250):
    paras = [p.strip() for p in re.split(r"\\n\\s*\\n+", text) if p.strip()]
    chunk = ""
    for p in paras:
        if len(chunk) + len(p) + 2 <= max_chars:
            chunk = (chunk + "\\n\\n" + p).strip()
        else:
            if len(chunk) >= min_chars:
                yield chunk
                chunk = p
            else:
                chunk = (chunk + "\\n\\n" + p).strip()
                if len(chunk) >= min_chars:
                    yield chunk
                    chunk = ""
    if chunk:
        yield chunk

def build_datasets(pdf_dir: str, out_csv: str, chunks_csv: str = None):
    records = []
    chunk_records = []
    pdf_paths = sorted([p for p in Path(pdf_dir).glob("**/*.pdf")])

    for p in tqdm(pdf_paths, desc="Processing PDFs"):
        raw = extract_text_from_pdf(p)
        if not raw:
            continue
        title, abstract, results, conclusion, full_text = split_into_sections(raw)
        paper_id = p.stem

        records.append({
            "paper_id": paper_id,
            "filename": str(p),
            "title": title,
            "abstract": abstract,
            "results": results,
            "conclusion": conclusion,
            "full_text": full_text
        })

        trimmix = "\\n\\n".join([s for s in [abstract, results, conclusion] if s]).strip()
        base_for_chunks = trimmix if trimmix else full_text
        for i, ch in enumerate(paragraph_chunks(base_for_chunks)):
            chunk_records.append({
                "paper_id": paper_id,
                "chunk_id": f"{paper_id}_{i:03d}",
                "text": ch
            })

    # Create output directories if they don't exist
    out_csv_path = Path(out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    if chunks_csv:
        chunks_csv_path = Path(chunks_csv)
        chunks_csv_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame.from_records(records).to_csv(out_csv, index=False)
    if chunks_csv and len(chunk_records) > 0:
        pd.DataFrame.from_records(chunk_records).to_csv(chunks_csv, index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", type=str, required=True, help="Folder containing PDFs")
    ap.add_argument("--out_csv", type=str, default="./papers/csv_papers/spacebio_papers.csv")
    ap.add_argument("--chunks_csv", type=str, default="./papers/csv_chunks/spacebio_chunks.csv")
    args = ap.parse_args()
    build_datasets(args.pdf_dir, args.out_csv, args.chunks_csv)
