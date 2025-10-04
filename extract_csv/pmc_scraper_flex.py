
import re
import time
import argparse
from urllib.parse import urlparse
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

HDRS = {
    "User-Agent": "Mozilla/5.0 (RAG-spacebio/1.0; +https://example.com)"
}

def clean_text(txt: str) -> str:
    if not txt:
        return ""
    return re.sub(r"\s+", " ", txt).strip()

def normalize_pmc_url(u: str) -> str:
    """Accept both pmc.ncbi.nlm.nih.gov/articles/PMCxxxx/ and
       www.ncbi.nlm.nih.gov/pmc/articles/PMCxxxx/ ; return canonical http URL."""
    u = (u or "").strip()
    if not u:
        return ""
    # Ensure scheme
    if not re.match(r"^https?://", u):
        u = "https://" + u
    # Harmonize host to pmc.ncbi.nlm.nih.gov
    m = re.search(r"/(pmc/)?articles/(PMC[0-9]+)/?", u)
    if not m:
        return ""
    pmc_id = m.group(2)
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmc_id}/"

def extract_sections_from_pmc_html(html: str, base="https://pmc.ncbi.nlm.nih.gov"):
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # PDF URL
    pdf_url = ""
    link_pdf = soup.find("a", string=re.compile(r"\bPDF\b", re.I))
    if link_pdf and link_pdf.has_attr("href"):
        href = link_pdf["href"]
        if href.startswith("http"):
            pdf_url = href
        else:
            pdf_url = requests.compat.urljoin(base, href)
    if not pdf_url:
        for l in soup.find_all("link"):
            if l.get("type") == "application/pdf" and l.get("href"):
                href = l["href"]
                pdf_url = href if href.startswith("http") else requests.compat.urljoin(base, href)
                break

    def collect_section_text_by_heading(heading_regex):
        heading = None
        for tag in soup.find_all(re.compile("^h[1-6]$")):
            if re.search(heading_regex, tag.get_text(" ", strip=True), re.I):
                heading = tag
                break
        if not heading:
            return ""
        texts = []
        for sib in heading.find_all_next():
            # stop at next heading
            if sib.name and re.match(r"^h[1-6]$", sib.name) and sib is not heading:
                break
            if sib.name in ["p", "li"]:
                texts.append(sib.get_text(" ", strip=True))
        return clean_text(" ".join(texts))

    abstract = collect_section_text_by_heading(r"^\s*abstract\s*$")
    results = collect_section_text_by_heading(r"^\s*results?\s*$")
    conclusion = collect_section_text_by_heading(r"^\s*conclusion[s]?|concluding remarks|summary\s*$")

    # Full text fallback
    main = soup.find(id="maincontent") or soup.find(id="maincontentcontainer") or soup
    paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    full_text = clean_text(" ".join(paras))

    return title, abstract, results, conclusion, full_text, pdf_url

def paragraph_chunks(text: str, max_chars=900, min_chars=250):
    # sentence-based accumulation
    pieces = re.split(r"(?<=[\.\?\!])\s+", text)
    chunk = ""
    for p in pieces:
        if len(chunk) + len(p) + 1 <= max_chars:
            chunk = (chunk + " " + p).strip()
        else:
            if len(chunk) >= min_chars:
                yield chunk
                chunk = p
            else:
                chunk = (chunk + " " + p).strip()
                if len(chunk) >= min_chars:
                    yield chunk
                    chunk = ""
    if chunk:
        yield chunk

def fetch(url: str, timeout=25) -> str:
    try:
        r = requests.get(url, headers=HDRS, timeout=timeout)
        if r.status_code == 200:
            return r.text
    except Exception:
        return ""
    return ""

def process(in_csv: str, out_papers: str, out_chunks: str, delay=0.7):
    df = pd.read_csv(in_csv)
    # Accept flexible column names
    col_url = None
    for c in df.columns:
        if c.lower().strip() in ("url","link","links"):
            col_url = c
            break
    if not col_url:
        raise ValueError("Input CSV must contain a 'url' or 'Link' column.")

    col_title = None
    for c in df.columns:
        if c.lower().strip() in ("title","name"):
            col_title = c
            break

    papers, chunks = [], []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scraping PMC"):
        raw_url = str(row[col_url]).strip()
        url = normalize_pmc_url(raw_url)
        if not url:
            # skip non-PMC
            continue

        html = fetch(url)
        if not html:
            continue

        parsed_title, abstract, results, conclusion, full_text, pdf_url = extract_sections_from_pmc_html(html, base="https://pmc.ncbi.nlm.nih.gov")
        input_title = str(row[col_title]).strip() if col_title else ""

        # paper_id
        m = re.search(r"/articles/(PMC[0-9]+)/", url)
        paper_id = m.group(1) if m else f"pmc_{i:05d}"

        final_title = parsed_title or input_title or paper_id

        papers.append({
            "paper_id": paper_id,
            "url": url,
            "title": final_title,
            "title_input": input_title,
            "pdf_url": pdf_url,
            "abstract": abstract,
            "results": results,
            "conclusion": conclusion,
            "full_text": full_text
        })

        # Build chunks from high-signal sections; fallback to full_text
        base = " ".join([s for s in [abstract, results, conclusion] if s]).strip()
        if not base:
            base = full_text
        idx = 0
        for ch in paragraph_chunks(base):
            chunks.append({
                "paper_id": paper_id,
                "chunk_id": f"{paper_id}_{idx:03d}",
                "text": ch
            })
            idx += 1

        time.sleep(delay)

    pd.DataFrame(papers).to_csv(out_papers, index=False)
    pd.DataFrame(chunks).to_csv(out_chunks, index=False)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV with 'url' or 'Link' column.")
    ap.add_argument("--out_papers", default="csv_papers.csv")
    ap.add_argument("--out_chunks", default="csv_chunks.csv")
    ap.add_argument("--delay", type=float, default=0.7)
    args = ap.parse_args()
    process(args.in_csv, args.out_papers, args.out_chunks, delay=args.delay)
