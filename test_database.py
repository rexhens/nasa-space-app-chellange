#!/usr/bin/env python3
"""
NASA Space App Challenge - Database Test Script
==============================================

Comprehensive test script to demonstrate all database functionality.

Usage:
    python test_database.py

Author: NASA Space App Challenge Team
"""

import sys
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

def test_database_comprehensive():
    """Test all database functionality comprehensively."""
    print("🚀 NASA Space App Challenge - Database Test")
    print("=" * 50)
    
    # Initialize database
    db_path = "space_research_test.db"
    db = SpaceResearchDB(db_path)
    
    print(f"Database initialized at: {db_path}")
    
    # Test 1: Get database statistics
    print("\n📊 Database Statistics:")
    stats = db.get_database_stats()
    print(f"  Papers: {stats['papers']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Queries: {stats['queries']}")
    print(f"  Database size: {stats['database_size_bytes']:,} bytes")
    
    # Test 2: Get all papers
    print("\n📄 All Papers:")
    papers = db.get_all_papers()
    print(f"  Total papers: {len(papers)}")
    
    if papers:
        print("\n  Sample papers:")
        for i, paper in enumerate(papers[:3]):
            print(f"    {i+1}. {paper['title'][:60]}...")
            print(f"       ID: {paper['paper_id']}")
            print(f"       Created: {paper['created_at']}")
            print()
    
    # Test 3: Get all chunks
    print("📝 All Chunks:")
    chunks = db.get_all_chunks()
    print(f"  Total chunks: {len(chunks)}")
    
    if chunks:
        print("\n  Sample chunks:")
        for i, chunk in enumerate(chunks[:3]):
            print(f"    {i+1}. {chunk['chunk_id']}")
            print(f"       Paper: {chunk['paper_id']}")
            print(f"       Text: {chunk['text'][:80]}...")
            print()
    
    # Test 4: Get all queries
    print("🔍 All Queries:")
    queries = db.get_all_queries()
    print(f"  Total queries: {len(queries)}")
    
    if queries:
        print("\n  Sample queries:")
        for i, query in enumerate(queries[:3]):
            print(f"    {i+1}. Query: {query['query_text'][:50]}...")
            print(f"       Timestamp: {query['timestamp']}")
            print(f"       Answer: {query['generated_answer'][:50] if query['generated_answer'] else 'None'}...")
            print()
    
    # Test 5: Search for specific content
    print("🔎 Content Search Examples:")
    
    # Search for papers about microgravity
    microgravity_papers = [p for p in papers if (p['title'] and 'microgravity' in p['title'].lower()) or (p['abstract'] and 'microgravity' in p['abstract'].lower())]
    print(f"  Papers about microgravity: {len(microgravity_papers)}")
    
    # Search for papers about bone
    bone_papers = [p for p in papers if (p['title'] and 'bone' in p['title'].lower()) or (p['abstract'] and 'bone' in p['abstract'].lower())]
    print(f"  Papers about bone: {len(bone_papers)}")
    
    # Search for papers about plants
    plant_papers = [p for p in papers if (p['title'] and 'plant' in p['title'].lower()) or (p['abstract'] and 'plant' in p['abstract'].lower())]
    print(f"  Papers about plants: {len(plant_papers)}")
    
    # Test 6: Database integrity check
    print("\n🔧 Database Integrity Check:")
    
    # Check if all chunks have corresponding papers
    chunk_paper_ids = set(chunk['paper_id'] for chunk in chunks)
    paper_ids = set(paper['paper_id'] for paper in papers)
    
    orphaned_chunks = chunk_paper_ids - paper_ids
    if orphaned_chunks:
        print(f"  ⚠️  Found {len(orphaned_chunks)} orphaned chunks")
    else:
        print("  ✅ All chunks have corresponding papers")
    
    # Check if all queries have valid data
    valid_queries = [q for q in queries if q['query_text'] and q['query_text'].strip()]
    print(f"  ✅ Valid queries: {len(valid_queries)}/{len(queries)}")
    
    print("\n🎉 Database test completed successfully!")
    print(f"Database contains {stats['papers']} papers, {stats['chunks']} chunks, and {stats['queries']} queries")

if __name__ == "__main__":
    try:
        test_database_comprehensive()
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        sys.exit(1)
