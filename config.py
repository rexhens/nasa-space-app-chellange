#!/usr/bin/env python3
"""
NASA Space App Challenge - Configuration Module
===============================================

Centralized configuration for the space biology research RAG system.
All configurable parameters are defined here for easy maintenance.

Author: NASA Space App Challenge Team
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
PDF_DIR = PROJECT_ROOT / "papers" / "pdf_papers"
CSV_PAPERS_DIR = PROJECT_ROOT / "papers" / "csv_papers"
CSV_CHUNKS_DIR = PROJECT_ROOT / "papers" / "csv_chunks"
INDEX_DIR = PROJECT_ROOT / "space_index"
DB_PATH = PROJECT_ROOT / "space_research.db"

# Default file names
DEFAULT_PAPERS_CSV = CSV_PAPERS_DIR / "spacebio_papers.csv"
DEFAULT_CHUNKS_CSV = CSV_CHUNKS_DIR / "spacebio_chunks.csv"

# PDF Processing Configuration
PDF_CONFIG = {
    "max_chars_per_chunk": 900,
    "min_chars_per_chunk": 250,
    "section_patterns": [
        "abstract", "introduction", "background", 
        "methods", "materials", "results", 
        "discussion", "conclusion", "conclusions", "references"
    ],
    "title_max_lines": 12,
    "first_page_lines": 150
}

# RAG System Configuration
RAG_CONFIG = {
    "embedding_model": "all-mpnet-base-v2",
    "chunk_size": 2000,
    "chunk_overlap": 200,
    "top_k": 5,
    "batch_size": 64,
    "similarity_threshold": 0.7
}

# OpenAI Configuration
OPENAI_CONFIG = {
    "model": "gpt-4o-mini",
    "max_tokens": 512,
    "temperature": 0.0,
    "timeout": 30
}

# Database Configuration
DATABASE_CONFIG = {
    "path": str(DB_PATH),
    "backup_enabled": True,
    "backup_interval": 24,  # hours
    "max_query_history": 1000
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": PROJECT_ROOT / "logs" / "space_research.log",
    "max_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "max_concurrent_pdfs": 4,
    "embedding_cache_size": 1000,
    "index_rebuild_threshold": 0.1,  # 10% of documents changed
    "memory_limit_mb": 2048
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dictionary containing all configuration parameters
    """
    return {
        "project_root": PROJECT_ROOT,
        "pdf_dir": PDF_DIR,
        "csv_papers_dir": CSV_PAPERS_DIR,
        "csv_chunks_dir": CSV_CHUNKS_DIR,
        "index_dir": INDEX_DIR,
        "db_path": DB_PATH,
        "pdf_config": PDF_CONFIG,
        "rag_config": RAG_CONFIG,
        "openai_config": OPENAI_CONFIG,
        "database_config": DATABASE_CONFIG,
        "logging_config": LOGGING_CONFIG,
        "performance_config": PERFORMANCE_CONFIG
    }

def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        PDF_DIR,
        CSV_PAPERS_DIR,
        CSV_CHUNKS_DIR,
        INDEX_DIR,
        LOGGING_CONFIG["file"].parent
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def validate_config() -> bool:
    """
    Validate the configuration settings.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Check if required directories exist or can be created
        ensure_directories()
        
        # Validate numeric parameters
        assert PDF_CONFIG["max_chars_per_chunk"] > PDF_CONFIG["min_chars_per_chunk"]
        assert RAG_CONFIG["chunk_size"] > RAG_CONFIG["chunk_overlap"]
        assert RAG_CONFIG["top_k"] > 0
        assert OPENAI_CONFIG["max_tokens"] > 0
        assert OPENAI_CONFIG["temperature"] >= 0.0
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("Configuration loaded successfully!")
    print(f"Project root: {config['project_root']}")
    print(f"PDF directory: {config['pdf_dir']}")
    print(f"Database path: {config['db_path']}")
    
    if validate_config():
        print("Configuration validation passed!")
    else:
        print("Configuration validation failed!")
