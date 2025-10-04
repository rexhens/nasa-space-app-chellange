#!/usr/bin/env python3
"""
NASA Space App Challenge - Database Reset Script for PMC Data
============================================================

Clear existing database and populate with only PMC extracted data.

Usage:
    python reset_database_pmc.py

Author: NASA Space App Challenge Team
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the database class
import importlib.util
spec = importlib.util.spec_from_file_location("open_ai_client", project_root / "First-working-prototype" / "open_ai_client.py")
open_ai_client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(open_ai_client)
SpaceResearchDB = open_ai_client.SpaceResearchDB

def clear_database(db_path):
    """Clear all data from the database."""
    print(f"🗑️  Clearing database: {db_path}")
    
    # Remove the database file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Deleted existing database file")
    else:
        print(f"ℹ️  Database file doesn't exist, creating new one")

def populate_with_pmc_data(db_path):
    """Populate database with PMC extracted data."""
    print(f"📚 Populating database with PMC data...")
    
    # Initialize new database
    db = SpaceResearchDB(db_path)
    
    # Paths to PMC extracted files
    papers_csv = project_root / "papers" / "csv_papers" / "pmc_extracted_papers_full.csv"
    chunks_csv = project_root / "papers" / "csv_chunks" / "pmc_extracted_chunks_full.csv"
    
    # Check if files exist
    if not papers_csv.exists():
        print(f"❌ Papers CSV not found: {papers_csv}")
        return False
    
    if not chunks_csv.exists():
        print(f"❌ Chunks CSV not found: {chunks_csv}")
        return False
    
    # Insert papers
    print(f"📄 Inserting papers from: {papers_csv}")
    papers_count = db.insert_papers_from_csv(str(papers_csv))
    
    # Insert chunks
    print(f"📝 Inserting chunks from: {chunks_csv}")
    chunks_count = db.insert_chunks_from_csv(str(chunks_csv))
    
    print(f"✅ Successfully inserted:")
    print(f"   - {papers_count} papers")
    print(f"   - {chunks_count} chunks")
    
    return True

def verify_database(db_path):
    """Verify the database contents."""
    print(f"🔍 Verifying database contents...")
    
    db = SpaceResearchDB(db_path)
    stats = db.get_database_stats()
    
    print(f"\n📊 Database Statistics:")
    print(f"   Papers: {stats['papers']}")
    print(f"   Chunks: {stats['chunks']}")
    print(f"   Queries: {stats['queries']}")
    print(f"   Database size: {stats['database_size_bytes']:,} bytes")
    
    # Show sample papers
    papers = db.get_all_papers()
    if papers:
        print(f"\n📄 Sample Papers:")
        for i, paper in enumerate(papers[:5], 1):
            print(f"   {i}. {paper['title'][:60]}...")
            print(f"      ID: {paper['paper_id']}")
            print(f"      URL: {paper.get('url', 'N/A')}")
            print()
    
    # Show sample chunks
    chunks = db.get_all_chunks()
    if chunks:
        print(f"📝 Sample Chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"   {i}. {chunk['chunk_id']}")
            print(f"      Paper: {chunk['paper_id']}")
            print(f"      Text: {chunk['text'][:80]}...")
            print()
    
    return True

def main():
    """Main function to reset and populate database."""
    print("🚀 NASA Space App Challenge - Database Reset for PMC Data")
    print("=" * 60)
    
    # Database path
    db_path = "space_research_test.db"
    
    try:
        # Step 1: Clear existing database
        clear_database(db_path)
        
        # Step 2: Populate with PMC data
        if not populate_with_pmc_data(db_path):
            print("❌ Failed to populate database")
            return 1
        
        # Step 3: Verify database
        if not verify_database(db_path):
            print("❌ Database verification failed")
            return 1
        
        print("\n🎉 Database reset completed successfully!")
        print("📚 Database now contains only PMC extracted papers")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during database reset: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
