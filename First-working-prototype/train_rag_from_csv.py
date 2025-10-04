#!/usr/bin/env python3
"""
train_rag_from_csv.py

Usage:
  1) Prepare CSV with columns: author, abstract, intro, background, methods, discussions, conclusions
  2) Set environment variable OPENAI_API_KEY if you want the script to use OpenAI for generation.
  3) Run:
       python train_rag_from_csv.py --csv mydata.csv --index_dir ./space_index

This will build embeddings + a FAISS index in index_dir. Use the `--query` flag to run interactive queries.
"""

import argparse
import os
import json
import math
from pathlib import Path
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# embeddings and index
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# optional generation with OpenAI
try:
    import openai
except Exception:
    openai = None

# ---------- Configurable parameters ----------
EMBED_MODEL = "all-mpnet-base-v2"   # small & strong general-purpose embedder
CHUNK_SIZE = 2000                   # characters per chunk (approx; tune to ~500-1200 tokens)
CHUNK_OVERLAP = 200                 # overlap between chunks
TOP_K = 5                           # number of retrieved chunks for answer generation
# --------------------------------------------

def stitch_row_to_text(row: pd.Series, fields=None) -> str:
    if fields is None:
        fields = ["author", "abstract", "intro", "background", "methods", "discussions", "conclusions"]
    parts = []
    for f in fields:
        if f in row and pd.notna(row[f]):
            header = f.upper()
            parts.append(f"{header}\n{row[f].strip()}")
    return "\n\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

def build_documents_from_csv(csv_path: str, text_fields=None):
    df = pd.read_csv(csv_path)
    docs = []
    for idx, row in df.iterrows():
        doc_text = stitch_row_to_text(row, fields=text_fields)
        if not doc_text:
            continue
        chunks = chunk_text(doc_text)
        for i, c in enumerate(chunks):
            docs.append({
                "doc_id": f"row{idx}",
                "chunk_id": f"row{idx}_chunk{i}",
                "text": c,
                "meta": {
                    "source_row": int(idx),
                    "author": row.get("author", ""),
                    "title": row.get("title", "") if "title" in row else "",
                    "original_fields": text_fields or []
                }
            })
    return docs

def embed_texts(texts: List[str], model_name=EMBED_MODEL, batch_size=64):
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
    # normalize for cosine similarity in FAISS (IndexFlatIP)
    faiss.normalize_L2(embs)
    return embs, model

def build_faiss_index(embs: np.ndarray):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product after normalization -> cosine
    index.add(embs)
    return index

def save_index_and_metadata(index, embs, docs, index_dir: str):
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    # Save embeddings (optional)
    np.save(os.path.join(index_dir, "embeddings.npy"), embs)
    # Save docs metadata
    with open(os.path.join(index_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Saved index and metadata to {index_dir}")

def load_index_and_metadata(index_dir: str):
    idxfile = os.path.join(index_dir, "faiss.index")
    docsfile = os.path.join(index_dir, "docs.jsonl")
    if not os.path.exists(idxfile) or not os.path.exists(docsfile):
        raise FileNotFoundError("Index files not found in " + index_dir)
    index = faiss.read_index(idxfile)
    docs = [json.loads(line) for line in open(docsfile, "r", encoding="utf-8")]
    return index, docs

def retrieve(index, embs, docs, query_emb: np.ndarray, top_k=TOP_K):
    # query_emb must be normalized already
    D, I = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(docs):
            continue
        results.append({"score": float(score), "doc": docs[idx]})
    return results

def generate_answer_with_openai(question: str, retrieved: List[dict], openai_model="gpt-4o-mini", max_tokens=512):
    if openai is None:
        raise RuntimeError("openai package not installed or available.")
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Build a context with sources
    context = ""
    for r in retrieved:
        s = r["doc"]
        context += f"Source: {s['doc_id']}/{s['chunk_id']} (author={s['meta'].get('author','')})\n{s['text']}\n\n---\n\n"
    prompt = (
        "You are a helpful domain expert. Use ONLY the provided source passages to answer the question. "
        "Cite the source ids in your answer where appropriate.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer (be concise, include source citations e.g. [row12_chunk0]):"
    )
    
    resp = client.chat.completions.create(
        model=openai_model,
        messages=[{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    out = resp.choices[0].message.content
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", help="Path to CSV dataset")
    p.add_argument("--index_dir", default="./index_data", help="Directory to save index & metadata")
    p.add_argument("--build", action="store_true", help="Build embeddings & index from CSV")
    p.add_argument("--query", type=str, help="Run a single query against saved index")
    p.add_argument("--top_k", type=int, default=TOP_K, help="Number of retrieved chunks to return")
    p.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model for generation (if using OpenAI)")
    args = p.parse_args()

    if args.build:
        if not args.csv:
            print("Error: --csv is required when using --build")
            return
        print("Reading CSV and building docs...")
        docs = build_documents_from_csv(args.csv)
        texts = [d["text"] for d in docs]
        print(f"Built {len(texts)} chunks from CSV")
        embs, model = embed_texts(texts)
        index = build_faiss_index(embs)
        save_index_and_metadata(index, embs, docs, args.index_dir)
        print("Index built successfully.")
        return

    if args.query:
        # load index & docs
        index, docs = load_index_and_metadata(args.index_dir)
        # reuse same embedder
        model = SentenceTransformer(EMBED_MODEL)
        q_emb = model.encode([args.query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, args.top_k)
        retrieved = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(docs): continue
            retrieved.append({"score": float(score), "doc": docs[idx]})
        print("=== Retrieved passages ===")
        for r in retrieved:
            print(f"[score={r['score']:.4f}] {r['doc']['doc_id']} / {r['doc']['chunk_id']} (author={r['doc']['meta'].get('author','')})")
            print(r['doc']['text'][:400].replace("\n"," ") + ("..." if len(r['doc']['text'])>400 else ""))
            print("----")
        # generate with OpenAI if configured
        if os.getenv("OPENAI_API_KEY") and openai is not None:
            answer = generate_answer_with_openai(args.query, retrieved, openai_model=args.openai_model)
            print("\n=== Generated answer ===\n")
            print(answer)
        else:
            print("\nNo OpenAI key found or openai package missing. You can still use the retrieved passages locally.")
        return

    p.print_help()

if __name__ == "__main__":
    main()
