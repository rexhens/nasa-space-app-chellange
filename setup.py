#!/usr/bin/env python3
"""
NASA Space App Challenge - Setup Script
======================================

Automated setup script for the space biology research RAG system.

Usage:
    python setup.py

Author: NASA Space App Challenge Team
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "First-working-prototype" / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Requirements file not found")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "papers/pdf_papers",
        "papers/csv_papers", 
        "papers/csv_chunks",
        "space_index",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment():
    """Set up environment configuration."""
    print("Setting up environment...")
    
    env_template = Path(__file__).parent / "First-working-prototype" / "env.template"
    env_file = Path(__file__).parent / ".env"
    
    if env_template.exists() and not env_file.exists():
        shutil.copy(env_template, env_file)
        print("✓ Environment file created")
        print("⚠️  Please edit .env file with your OpenAI API key")
    else:
        print("✓ Environment file already exists")

def test_installation():
    """Test the installation."""
    print("Testing installation...")
    
    try:
        # Test imports
        import pandas
        import sentence_transformers
        import faiss
        import numpy
        import tqdm
        import fitz  # PyMuPDF
        
        print("✓ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function."""
    print("🚀 NASA Space App Challenge - Setup")
    print("=" * 50)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Installing dependencies", install_dependencies),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return 1
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your OpenAI API key")
    print("2. Place PDF files in papers/pdf_papers/")
    print("3. Run: python pdf_to_csv/pdf_to_csv.py --pdf_dir papers/pdf_papers")
    print("4. Run: python First-working-prototype/rag.py --csv papers/csv_papers/spacebio_papers.csv --build")
    print("5. Test with: python test_system.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
