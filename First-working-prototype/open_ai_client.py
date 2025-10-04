#!/usr/bin/env python3
"""
NASA Space App Challenge - RAG System with Database Integration
==============================================================

A Retrieval-Augmented Generation (RAG) system for space biology research papers.
Supports both CSV-based and database-stored data with FAISS indexing.

Usage:
  1) Process PDFs: python pdf_to_csv.py --pdf_dir ./papers/pdf_papers
  2) Build RAG index: python rag.py --csv papers/csv_papers/spacebio_papers.csv --build
  3) Query system: python rag.py --query "What are the effects of microgravity on bone density?"

Features:
- FAISS vector indexing for fast similarity search
- Database integration for persistent storage
- OpenAI integration for answer generation
- Configurable embedding models and parameters

Author: NASA Space App Challenge Team
"""

import argparse
import os
import json
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Embeddings and indexing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Optional OpenAI integration
try:
    import openai
except ImportError:
    openai = None
    logger.warning("OpenAI package not installed. Answer generation will be disabled.")

# ---------- Configurable parameters ----------
EMBED_MODEL = "all-mpnet-base-v2"   # small & strong general-purpose embedder
CHUNK_SIZE = 2000                   # characters per chunk (approx; tune to ~500-1200 tokens)
CHUNK_OVERLAP = 200                 # overlap between chunks
TOP_K = 5                           # number of retrieved chunks for answer generation
DB_PATH = "./space_research.db"      # SQLite database path
# --------------------------------------------

class SpaceResearchDB:
    """Database interface for storing and retrieving space research data."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Papers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    filename TEXT,
                    title TEXT,
                    abstract TEXT,
                    results TEXT,
                    conclusion TEXT,
                    full_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Chunks table for embeddings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    text TEXT,
                    embedding_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
                )
            """)
            
            # Queries table for tracking usage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    query_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_text TEXT,
                    retrieved_chunks TEXT,
                    generated_answer TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def insert_papers_from_csv(self, csv_path: str) -> int:
        """Insert papers from CSV into database."""
        df = pd.read_csv(csv_path)
        inserted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    # Handle different CSV formats (PMC vs PDF extracted)
                    filename = row.get('filename', row.get('url', ''))
                    
                    cursor.execute("""
                        INSERT OR REPLACE INTO papers 
                        (paper_id, filename, title, abstract, results, conclusion, full_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['paper_id'],
                        filename,
                        row['title'],
                        row['abstract'],
                        row['results'],
                        row['conclusion'],
                        row['full_text']
                    ))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting paper {row['paper_id']}: {e}")
            
            conn.commit()
        
        logger.info(f"Inserted {inserted_count} papers into database")
        return inserted_count
    
    def insert_chunks_from_csv(self, csv_path: str) -> int:
        """Insert chunks from CSV into database."""
        df = pd.read_csv(csv_path)
        inserted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (chunk_id, paper_id, text, embedding_id)
                        VALUES (?, ?, ?, ?)
                    """, (
                        row['chunk_id'],
                        row['paper_id'],
                        row['text'],
                        inserted_count  # This will be updated when embeddings are created
                    ))
                    inserted_count += 1
                except Exception as e:
                    logger.error(f"Error inserting chunk {row['chunk_id']}: {e}")
            
            conn.commit()
        
        logger.info(f"Inserted {inserted_count} chunks into database")
        return inserted_count
    
    def get_all_chunks(self) -> List[Dict]:
        """Retrieve all chunks from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT chunk_id, paper_id, text FROM chunks")
            return [{"chunk_id": row[0], "paper_id": row[1], "text": row[2]} for row in cursor.fetchall()]
    
    def get_all_papers(self) -> List[Dict]:
        """Retrieve all papers from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT paper_id, filename, title, abstract, results, conclusion, created_at FROM papers")
            return [{
                "paper_id": row[0], 
                "filename": row[1], 
                "title": row[2], 
                "abstract": row[3], 
                "results": row[4], 
                "conclusion": row[5], 
                "created_at": row[6]
            } for row in cursor.fetchall()]
    
    def get_all_queries(self) -> List[Dict]:
        """Retrieve all queries from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT query_id, query_text, retrieved_chunks, generated_answer, timestamp FROM queries")
            return [{
                "query_id": row[0], 
                "query_text": row[1], 
                "retrieved_chunks": row[2], 
                "generated_answer": row[3], 
                "timestamp": row[4]
            } for row in cursor.fetchall()]
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Count papers
            cursor.execute("SELECT COUNT(*) FROM papers")
            paper_count = cursor.fetchone()[0]
            
            # Count chunks
            cursor.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            # Count queries
            cursor.execute("SELECT COUNT(*) FROM queries")
            query_count = cursor.fetchone()[0]
            
            # Get database size
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            
            return {
                "papers": paper_count,
                "chunks": chunk_count,
                "queries": query_count,
                "database_size_bytes": db_size,
                "database_path": self.db_path
            }
    
    def log_query(self, query_text: str, retrieved_chunks: List[Dict], answer: Optional[str] = None):
        """Log a query and its results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO queries (query_text, retrieved_chunks, generated_answer)
                VALUES (?, ?, ?)
            """, (
                query_text,
                json.dumps(retrieved_chunks),
                answer
            ))
            conn.commit()

def stitch_row_to_text(row: pd.Series, fields: Optional[List[str]] = None) -> str:
    """
    Combine multiple fields from a CSV row into a single text string.
    
    Args:
        row: Pandas Series representing a CSV row
        fields: List of field names to include (defaults to space research fields)
        
    Returns:
        Combined text string
    """
    if fields is None:
        fields = ["title", "abstract", "results", "conclusion"]
    
    parts = []
    for field in fields:
        if field in row and pd.notna(row[field]) and str(row[field]).strip():
            header = field.upper()
            parts.append(f"{header}\n{str(row[field]).strip()}")
    
    return "\n\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks for embedding.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
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

def build_documents_from_csv(csv_path: str, text_fields: Optional[List[str]] = None) -> List[Dict]:
    """
    Build document chunks from CSV data for RAG system.
    
    Args:
        csv_path: Path to CSV file
        text_fields: Fields to include in document text
        
    Returns:
        List of document dictionaries
    """
    df = pd.read_csv(csv_path)
    docs = []
    
    logger.info(f"Processing {len(df)} rows from CSV")
    
    for idx, row in df.iterrows():
        try:
            doc_text = stitch_row_to_text(row, fields=text_fields)
            if not doc_text:
                logger.warning(f"No text content for row {idx}")
                continue
            
            chunks = chunk_text(doc_text)
            for i, chunk in enumerate(chunks):
                docs.append({
                    "doc_id": f"row{idx}",
                    "chunk_id": f"row{idx}_chunk{i}",
                    "text": chunk,
                    "meta": {
                        "source_row": int(idx),
                        "paper_id": row.get("paper_id", ""),
                        "title": row.get("title", ""),
                        "filename": row.get("filename", ""),
                        "original_fields": text_fields or []
                    }
                })
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue
    
    logger.info(f"Built {len(docs)} document chunks")
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
    """Main entry point for the RAG system."""
    parser = argparse.ArgumentParser(
        description="NASA Space Research RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from CSV
  python rag.py --csv papers/csv_papers/spacebio_papers.csv --build
  
  # Query the system
  python rag.py --query "What are the effects of microgravity on bone density?"
  
  # Use database integration
  python rag.py --csv papers/csv_papers/spacebio_papers.csv --build --use_db
        """
    )
    
    parser.add_argument("--csv", help="Path to CSV dataset")
    parser.add_argument("--chunks_csv", help="Path to chunks CSV dataset")
    parser.add_argument("--index_dir", default="./space_index", help="Directory to save index & metadata")
    parser.add_argument("--build", action="store_true", help="Build embeddings & index from CSV")
    parser.add_argument("--query", type=str, help="Run a single query against saved index")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of retrieved chunks to return")
    parser.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="OpenAI model for generation")
    parser.add_argument("--use_db", action="store_true", help="Use database integration")
    parser.add_argument("--db_path", type=str, default=DB_PATH, help="Database path")
    parser.add_argument("--test_db", action="store_true", help="Test database functionality")
    
    args = parser.parse_args()

    # Initialize database if requested
    db = None
    if args.use_db or args.test_db:
        db = SpaceResearchDB(args.db_path)

    if args.test_db:
        logger.info("Testing database functionality...")
        
        # Get database statistics
        stats = db.get_database_stats()
        print("\n=== Database Statistics ===")
        print(f"Papers: {stats['papers']}")
        print(f"Chunks: {stats['chunks']}")
        print(f"Queries: {stats['queries']}")
        print(f"Database size: {stats['database_size_bytes']:,} bytes")
        print(f"Database path: {stats['database_path']}")
        
        # Get all papers
        papers = db.get_all_papers()
        print(f"\n=== Papers ({len(papers)} total) ===")
        for i, paper in enumerate(papers[:5]):  # Show first 5 papers
            print(f"{i+1}. {paper['title'][:80]}...")
            print(f"   ID: {paper['paper_id']}")
            print(f"   Created: {paper['created_at']}")
            print()
        
        if len(papers) > 5:
            print(f"... and {len(papers) - 5} more papers")
        
        # Get all chunks
        chunks = db.get_all_chunks()
        print(f"\n=== Chunks ({len(chunks)} total) ===")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"{i+1}. {chunk['chunk_id']}")
            print(f"   Paper: {chunk['paper_id']}")
            print(f"   Text: {chunk['text'][:100]}...")
            print()
        
        if len(chunks) > 3:
            print(f"... and {len(chunks) - 3} more chunks")
        
        # Get all queries
        queries = db.get_all_queries()
        print(f"\n=== Queries ({len(queries)} total) ===")
        for i, query in enumerate(queries[:3]):  # Show first 3 queries
            print(f"{i+1}. Query: {query['query_text'][:60]}...")
            print(f"   Timestamp: {query['timestamp']}")
            print(f"   Answer: {query['generated_answer'][:100] if query['generated_answer'] else 'None'}...")
            print()
        
        if len(queries) > 3:
            print(f"... and {len(queries) - 3} more queries")
        
        logger.info("Database test completed successfully!")
        return 0

    if args.build:
        if not args.csv:
            logger.error("--csv is required when using --build")
            return 1
        
        logger.info("Building RAG index from CSV...")
        
        # Insert data into database if requested
        if db:
            logger.info("Inserting papers into database...")
            db.insert_papers_from_csv(args.csv)
            
            if args.chunks_csv and os.path.exists(args.chunks_csv):
                logger.info("Inserting chunks into database...")
                db.insert_chunks_from_csv(args.chunks_csv)
        
        # Build documents and embeddings
        docs = build_documents_from_csv(args.csv)
        texts = [d["text"] for d in docs]
        
        if not texts:
            logger.error("No text content found in CSV")
            return 1
        
        logger.info(f"Building embeddings for {len(texts)} chunks...")
        embs, model = embed_texts(texts)
        index = build_faiss_index(embs)
        save_index_and_metadata(index, embs, docs, args.index_dir)
        
        logger.info("Index built successfully!")
        return 0

    if args.query:
        try:
            # Load index & docs
            index, docs = load_index_and_metadata(args.index_dir)
            model = SentenceTransformer(EMBED_MODEL)
            
            # Encode query
            q_emb = model.encode([args.query], convert_to_numpy=True)
            faiss.normalize_L2(q_emb)
            
            # Search
            D, I = index.search(q_emb, args.top_k)
            retrieved = []
            
            for score, idx in zip(D[0], I[0]):
                if 0 <= idx < len(docs):
                    retrieved.append({"score": float(score), "doc": docs[idx]})
            
            # Display results
            print("=== Retrieved Passages ===")
            for r in retrieved:
                meta = r['doc']['meta']
                print(f"[score={r['score']:.4f}] {r['doc']['doc_id']} / {r['doc']['chunk_id']}")
                print(f"Paper: {meta.get('title', 'Unknown')}")
                print(f"Text: {r['doc']['text'][:400].replace(chr(10), ' ')}...")
                print("----")
            
            # Generate answer with OpenAI if available
            answer = None
            if os.getenv("OPENAI_API_KEY") and openai is not None:
                try:
                    answer = generate_answer_with_openai(args.query, retrieved, openai_model=args.openai_model)
                    print("\n=== Generated Answer ===\n")
                    print(answer)
                except Exception as e:
                    logger.error(f"OpenAI generation failed: {e}")
            else:
                print("\nNo OpenAI key found. Retrieved passages available above.")
            
            # Log query if database is available
            if db:
                db.log_query(args.query, retrieved, answer)
            
            return 0
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return 1

    parser.print_help()
    return 0

if __name__ == "__main__":
    exit(main())
