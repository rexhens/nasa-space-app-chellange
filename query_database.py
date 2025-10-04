#!/usr/bin/env python3
"""
NASA Space App Challenge - Database Query Script
===============================================

Interactive script to query the database for specific information.

Usage:
    python query_database.py

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

def search_papers_by_topic(db, topic):
    """Search for papers containing a specific topic."""
    papers = db.get_all_papers()
    matching_papers = []
    
    for paper in papers:
        # Handle None values safely
        title = paper['title'] or ""
        abstract = paper['abstract'] or ""
        results = paper['results'] or ""
        conclusion = paper['conclusion'] or ""
        
        if (topic.lower() in title.lower() or 
            topic.lower() in abstract.lower() or
            topic.lower() in results.lower() or
            topic.lower() in conclusion.lower()):
            matching_papers.append(paper)
    
    return matching_papers

def get_paper_details(db, paper_id):
    """Get detailed information about a specific paper."""
    papers = db.get_all_papers()
    for paper in papers:
        if paper['paper_id'] == paper_id:
            return paper
    return None

def get_chunks_for_paper(db, paper_id):
    """Get all chunks for a specific paper."""
    chunks = db.get_all_chunks()
    return [chunk for chunk in chunks if chunk['paper_id'] == paper_id]

def main():
    """Main interactive function."""
    print("🚀 NASA Space App Challenge - Database Query Tool")
    print("=" * 50)
    
    # Initialize database
    db_path = "space_research_test.db"
    db = SpaceResearchDB(db_path)
    
    print(f"Database loaded: {db_path}")
    
    # Show available topics
    papers = db.get_all_papers()
    print(f"\n📚 Available papers: {len(papers)}")
    
    # Common topics in space research
    topics = ["microgravity", "bone", "muscle", "plant", "radiation", "spaceflight", "astronaut", "ISS"]
    print("\n🔍 Common topics to search:")
    for i, topic in enumerate(topics, 1):
        count = len(search_papers_by_topic(db, topic))
        print(f"  {i}. {topic.capitalize()} ({count} papers)")
    
    # Interactive search
    while True:
        print("\n" + "="*50)
        print("Search options:")
        print("1. Search by topic")
        print("2. Get paper details")
        print("3. Get chunks for a paper")
        print("4. Show database stats")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            topic = input("Enter topic to search: ").strip()
            if topic:
                matching_papers = search_papers_by_topic(db, topic)
                print(f"\n📄 Found {len(matching_papers)} papers about '{topic}':")
                for i, paper in enumerate(matching_papers[:5], 1):
                    print(f"  {i}. {paper['title'][:60]}...")
                    print(f"     ID: {paper['paper_id']}")
                if len(matching_papers) > 5:
                    print(f"     ... and {len(matching_papers) - 5} more")
        
        elif choice == "2":
            paper_id = input("Enter paper ID: ").strip()
            if paper_id:
                paper = get_paper_details(db, paper_id)
                if paper:
                    print(f"\n📄 Paper Details:")
                    print(f"  Title: {paper['title']}")
                    print(f"  ID: {paper['paper_id']}")
                    print(f"  Filename: {paper['filename']}")
                    print(f"  Created: {paper['created_at']}")
                    print(f"  Abstract: {paper['abstract'][:200]}...")
                    print(f"  Results: {paper['results'][:200]}...")
                    print(f"  Conclusion: {paper['conclusion'][:200]}...")
                else:
                    print("❌ Paper not found")
        
        elif choice == "3":
            paper_id = input("Enter paper ID: ").strip()
            if paper_id:
                chunks = get_chunks_for_paper(db, paper_id)
                print(f"\n📝 Found {len(chunks)} chunks for paper '{paper_id}':")
                for i, chunk in enumerate(chunks[:3], 1):
                    print(f"  {i}. {chunk['chunk_id']}")
                    print(f"     Text: {chunk['text'][:100]}...")
                if len(chunks) > 3:
                    print(f"     ... and {len(chunks) - 3} more chunks")
        
        elif choice == "4":
            stats = db.get_database_stats()
            print(f"\n📊 Database Statistics:")
            print(f"  Papers: {stats['papers']}")
            print(f"  Chunks: {stats['chunks']}")
            print(f"  Queries: {stats['queries']}")
            print(f"  Database size: {stats['database_size_bytes']:,} bytes")
        
        elif choice == "5":
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
