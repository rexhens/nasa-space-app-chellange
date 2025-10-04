#!/usr/bin/env python3
"""
NASA Space App Challenge - System Test Script
============================================

Test script to verify the complete PDF to RAG pipeline works correctly.

Usage:
    python test_system.py

Author: NASA Space App Challenge Team
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import get_config, validate_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration():
    """Test configuration loading and validation."""
    logger.info("Testing configuration...")
    
    config = get_config()
    if not validate_config():
        logger.error("Configuration validation failed!")
        return False
    
    logger.info("✓ Configuration loaded and validated")
    return True

def test_pdf_processing():
    """Test PDF processing functionality."""
    logger.info("Testing PDF processing...")
    
    config = get_config()
    pdf_dir = config["pdf_dir"]
    
    if not pdf_dir.exists():
        logger.error(f"PDF directory does not exist: {pdf_dir}")
        return False
    
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error("No PDF files found in PDF directory")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Test PDF processing
    try:
        cmd = [
            sys.executable, 
            "pdf_to_csv/pdf_to_csv.py",
            "--pdf_dir", str(pdf_dir),
            "--out_csv", str(config["csv_papers_dir"] / "test_papers.csv"),
            "--chunks_csv", str(config["csv_chunks_dir"] / "test_chunks.csv")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            logger.error(f"PDF processing failed: {result.stderr}")
            return False
        
        logger.info("✓ PDF processing completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("PDF processing timed out")
        return False
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        return False

def test_rag_system():
    """Test RAG system functionality."""
    logger.info("Testing RAG system...")
    
    config = get_config()
    test_csv = config["csv_papers_dir"] / "test_papers.csv"
    
    if not test_csv.exists():
        logger.error("Test CSV file not found. Run PDF processing first.")
        return False
    
    try:
        # Test index building
        cmd = [
            sys.executable,
            "First-working-prototype/rag.py",
            "--csv", str(test_csv),
            "--build",
            "--index_dir", str(config["index_dir"] / "test_index")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"RAG index building failed: {result.stderr}")
            return False
        
        logger.info("✓ RAG index built successfully")
        
        # Test query
        cmd = [
            sys.executable,
            "First-working-prototype/rag.py",
            "--query", "What are the effects of spaceflight?",
            "--index_dir", str(config["index_dir"] / "test_index")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            logger.error(f"RAG query failed: {result.stderr}")
            return False
        
        logger.info("✓ RAG query completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("RAG system test timed out")
        return False
    except Exception as e:
        logger.error(f"RAG system test failed: {e}")
        return False

def test_database_integration():
    """Test database integration."""
    logger.info("Testing database integration...")
    
    config = get_config()
    test_csv = config["csv_papers_dir"] / "test_papers.csv"
    
    if not test_csv.exists():
        logger.error("Test CSV file not found. Run PDF processing first.")
        return False
    
    try:
        cmd = [
            sys.executable,
            "First-working-prototype/rag.py",
            "--csv", str(test_csv),
            "--build",
            "--use_db",
            "--db_path", str(config["db_path"]).replace(".db", "_test.db")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode != 0:
            logger.error(f"Database integration test failed: {result.stderr}")
            return False
        
        logger.info("✓ Database integration test completed successfully")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Database integration test timed out")
        return False
    except Exception as e:
        logger.error(f"Database integration test failed: {e}")
        return False

def cleanup_test_files():
    """Clean up test files."""
    logger.info("Cleaning up test files...")
    
    config = get_config()
    test_files = [
        config["csv_papers_dir"] / "test_papers.csv",
        config["csv_chunks_dir"] / "test_chunks.csv",
        config["index_dir"] / "test_index",
        config["db_path"].parent / "space_research_test.db"
    ]
    
    for file_path in test_files:
        if file_path.exists():
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                import shutil
                shutil.rmtree(file_path)
    
    logger.info("✓ Test files cleaned up")

def main():
    """Run all system tests."""
    logger.info("Starting NASA Space App Challenge system tests...")
    
    tests = [
        ("Configuration", test_configuration),
        ("PDF Processing", test_pdf_processing),
        ("RAG System", test_rag_system),
        ("Database Integration", test_database_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test PASSED")
            else:
                logger.error(f"✗ {test_name} test FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} test FAILED with exception: {e}")
    
    logger.info(f"\n--- Test Results ---")
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    finally:
        cleanup_test_files()
    
    sys.exit(exit_code)
