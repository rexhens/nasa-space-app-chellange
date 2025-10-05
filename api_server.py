#!/usr/bin/env python3
"""
NASA Space App Challenge - RAG API Server
==========================================

Flask API server to expose the RAG system for testing with Postman.

Endpoints:
- POST /query - Query the RAG system
- GET /health - Health check
- GET /stats - Database statistics

Usage:
    python api_server.py

Author: NASA Space App Challenge Team
"""

import os
import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the RAG system
import importlib.util
spec = importlib.util.spec_from_file_location("open_ai_client", project_root / "First-working-prototype" / "open_ai_client.py")
open_ai_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(open_ai_client)

# Import database class
SpaceResearchDB = open_ai_client.SpaceResearchDB

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for RAG system
rag_system = None
db = None

def initialize_rag_system():
    """Initialize the RAG system components."""
    global rag_system, db
    
    try:
        # Initialize database
        db_path = "space_research_test.db"
        db = SpaceResearchDB(db_path)
        
        # Initialize RAG system components
        index_dir = "space_index/pmc_index_full"
        index, docs = open_ai_client.load_index_and_metadata(index_dir)
        rag_system = {
            'index_dir': index_dir,
            'index': index,
            'docs': docs
        }
        
        logger.info("RAG system initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "NASA Space App Challenge RAG API is running",
        "rag_system_initialized": rag_system is not None,
        "database_initialized": db is not None
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    try:
        if db is None:
            return jsonify({"error": "Database not initialized"}), 500
        
        stats = db.get_database_stats()
        return jsonify({
            "papers": stats['papers'],
            "chunks": stats['chunks'],
            "queries": stats['queries'],
            "database_size_bytes": stats['database_size_bytes'],
            "database_size_mb": round(stats['database_size_bytes'] / (1024 * 1024), 2)
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query_rag():
    """Query the RAG system."""
    try:
        if rag_system is None or db is None:
            return jsonify({"error": "RAG system not initialized"}), 500
        
        # Get request data
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request body"}), 400
        
        query_text = data['query']
        if not query_text.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        
        # Optional parameters
        top_k = data.get('top_k', 5)
        openai_model = data.get('model', 'gpt-3.5-turbo')
        
        logger.info(f"Processing query: {query_text}")
        
        # Retrieve relevant passages
        model = SentenceTransformer(open_ai_client.EMBED_MODEL)
        query_emb = model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        # Search using FAISS
        D, I = rag_system['index'].search(query_emb, top_k)
        retrieved = []
        
        for score, idx in zip(D[0], I[0]):
            if 0 <= idx < len(rag_system['docs']):
                retrieved.append({
                    "score": float(score), 
                    "doc": rag_system['docs'][idx]
                })
        
        # Generate answer using OpenAI
        answer = open_ai_client.generate_answer_with_openai(
            query_text, 
            retrieved, 
            openai_model=openai_model
        )
        
        # Log query to database
        db.log_query(query_text, retrieved, answer)
        
        # Format response
        response = {
            "query": query_text,
            "answer": answer,
            "retrieved_passages": [
                {
                    "score": passage['score'],
                    "paper_id": passage['doc']['meta'].get('paper_id', ''),
                    "chunk_id": passage['doc']['chunk_id'],
                    "title": passage['doc']['meta'].get('title', ''),
                    "text": passage['doc']['text'][:200] + "..." if len(passage['doc']['text']) > 200 else passage['doc']['text']
                }
                for passage in retrieved
            ],
            "total_passages": len(retrieved),
            "model_used": openai_model
        }
        
        logger.info(f"Query processed successfully: {len(retrieved)} passages retrieved")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/papers', methods=['GET'])
def get_papers():
    """Get list of papers with optional search."""
    try:
        if db is None:
            return jsonify({"error": "Database not initialized"}), 500
        
        # Get query parameters
        search = request.args.get('search', '')
        
        papers = db.get_all_papers()
        
        # Filter papers if search term provided
        if search:
            search_lower = search.lower()
            papers = [
                p for p in papers 
                if (p['title'] and search_lower in p['title'].lower()) or
                   (p['abstract'] and search_lower in p['abstract'].lower())
            ]
        
        # Format response
        response = {
            "papers": [
                {
                    "paper_id": p['paper_id'],
                    "title": p['title'],
                    "abstract": p['abstract'],
                    "created_at": p['created_at']
                }
                for p in papers
            ],
            "total_found": len(papers),
            "search_term": search
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting papers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/papers/<paper_id>', methods=['GET'])
def get_paper_details(paper_id):
    """Get detailed information about a specific paper."""
    try:
        if db is None:
            return jsonify({"error": "Database not initialized"}), 500
        
        papers = db.get_all_papers()
        paper = next((p for p in papers if p['paper_id'] == paper_id), None)
        
        if not paper:
            return jsonify({"error": f"Paper {paper_id} not found"}), 404
        
        # Get chunks for this paper
        chunks = db.get_all_chunks()
        paper_chunks = [c for c in chunks if c['paper_id'] == paper_id]
        
        response = {
            "paper_id": paper['paper_id'],
            "title": paper['title'],
            "abstract": paper['abstract'],
            "results": paper['results'],
            "conclusion": paper['conclusion'],
            "full_text": paper['full_text'],
            "created_at": paper['created_at'],
            "chunks": [
                {
                    "chunk_id": c['chunk_id'],
                    "text": c['text'][:200] + "..." if len(c['text']) > 200 else c['text']
                }
                for c in paper_chunks
            ],
            "total_chunks": len(paper_chunks)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error getting paper details: {e}")
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500

def main():
    """Main function to start the API server."""
    print("🚀 NASA Space App Challenge - RAG API Server")
    print("=" * 50)
    
    # Initialize RAG system
    if not initialize_rag_system():
        print("❌ Failed to initialize RAG system. Exiting.")
        sys.exit(1)
    
    print("✅ RAG system initialized successfully")
    print("📚 Database loaded with PMC papers")
    print("🌐 Starting API server...")
    print("\nAvailable endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /stats            - Database statistics")
    print("  POST /query            - Query the RAG system")
    print("  GET  /papers           - List papers (with optional search)")
    print("  GET  /papers/<id>      - Get paper details")
    print("\nExample POST /query request:")
    print('  {"query": "What are the effects of microgravity on bone health?"}')
    print("\nServer starting on http://localhost:8000")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=True)

if __name__ == "__main__":
    main()
