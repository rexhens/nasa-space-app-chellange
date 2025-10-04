# NASA Space App Challenge - Space Biology Research RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for analyzing space biology research papers, featuring PDF processing, database integration, and intelligent query answering.

## 🚀 Features

- **PDF Processing**: Extract structured data from research papers
- **Database Integration**: SQLite database for persistent storage
- **Vector Search**: FAISS indexing for fast similarity search
- **AI-Powered Answers**: OpenAI integration for intelligent responses
- **Comprehensive Logging**: Track queries and system performance

## 📁 Project Structure

```
nasa-space-app-challenge/
├── pdf_to_csv/                    # PDF processing module
│   └── pdf_to_csv.py             # Convert PDFs to structured CSV
├── First-working-prototype/       # RAG system implementation
│   ├── rag.py                    # Main RAG system with database
│   ├── requirements.txt           # Python dependencies
│   └── env.template              # Environment configuration
├── papers/                       # Research data
│   ├── pdf_papers/              # Original PDF files
│   ├── csv_papers/              # Processed paper data
│   └── csv_chunks/              # Text chunks for embeddings
└── README.md                     # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd nasa-space-app-challenge
   ```

2. **Install dependencies**
   ```bash
   pip install -r First-working-prototype/requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp First-working-prototype/env.template .env
   # Edit .env with your OpenAI API key
   ```

## 🚀 Quick Start

### Step 1: Process PDF Papers
```bash
python pdf_to_csv/pdf_to_csv.py --pdf_dir papers/pdf_papers
```

This will create:
- `papers/csv_papers/spacebio_papers.csv` - Structured paper data
- `papers/csv_chunks/spacebio_chunks.csv` - Text chunks for embeddings

### Step 2: Build RAG Index
```bash
python First-working-prototype/rag.py --csv papers/csv_papers/spacebio_papers.csv --build --use_db
```

### Step 3: Query the System
```bash
python First-working-prototype/rag.py --query "What are the effects of microgravity on bone density?"
```

## 📊 Data Processing Pipeline

### PDF to CSV Conversion
The `pdf_to_csv.py` script extracts:
- **Title**: Paper title
- **Abstract**: Research summary
- **Results**: Key findings
- **Conclusion**: Research conclusions
- **Full Text**: Complete paper content

### Database Schema
- **papers**: Main paper metadata and content
- **chunks**: Text chunks for vector search
- **queries**: Query history and results

### RAG System Components
1. **Document Processing**: Convert papers to searchable chunks
2. **Embedding Generation**: Create vector representations
3. **FAISS Indexing**: Build fast similarity search index
4. **Query Processing**: Retrieve relevant passages
5. **Answer Generation**: Generate AI-powered responses

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_MAX_TOKENS=512
```

### RAG Parameters
- **Embedding Model**: `all-mpnet-base-v2`
- **Chunk Size**: 2000 characters
- **Chunk Overlap**: 200 characters
- **Top K Results**: 5 passages

## 📝 Usage Examples

### Basic PDF Processing
```bash
# Process all PDFs in a directory
python pdf_to_csv/pdf_to_csv.py --pdf_dir ./papers/pdf_papers

# Custom output paths
python pdf_to_csv/pdf_to_csv.py \
  --pdf_dir ./papers/pdf_papers \
  --out_csv ./data/papers.csv \
  --chunks_csv ./data/chunks.csv
```

### RAG System Operations
```bash
# Build index with database integration
python rag.py --csv papers/csv_papers/spacebio_papers.csv --build --use_db

# Query without database
python rag.py --query "How does spaceflight affect muscle mass?"

# Query with custom parameters
python rag.py \
  --query "What countermeasures exist for bone loss in space?" \
  --top_k 10 \
  --openai_model gpt-4o
```

## 🔍 Query Examples

Try these research questions:
- "What are the effects of microgravity on bone density?"
- "How does spaceflight affect muscle mass and strength?"
- "What countermeasures exist for radiation exposure in space?"
- "How do plants grow in lunar regolith?"
- "What are the immunological effects of spaceflight?"

## 🛠️ Development

### Adding New Papers
1. Place PDF files in `papers/pdf_papers/`
2. Run the PDF processing script
3. Rebuild the RAG index

### Customizing the System
- **Embedding Model**: Change `EMBED_MODEL` in `rag.py`
- **Chunk Size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Database**: Modify `SpaceResearchDB` class for different storage

### Error Handling
The system includes comprehensive logging:
- PDF extraction errors
- Database operation failures
- OpenAI API issues
- Index building problems

## 📈 Performance

- **PDF Processing**: ~2-5 seconds per paper
- **Index Building**: ~1-2 minutes for 35 papers
- **Query Response**: ~2-5 seconds per query
- **Database Size**: ~50MB for 35 papers + chunks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the NASA Space App Challenge.

## 🆘 Troubleshooting

### Common Issues

**PDF Processing Fails**
- Ensure PyMuPDF is installed: `pip install PyMuPDF`
- Check PDF file permissions
- Verify PDF files are not corrupted

**Database Errors**
- Check SQLite installation
- Verify file permissions for database path
- Ensure sufficient disk space

**OpenAI API Issues**
- Verify API key is correct
- Check API quota and billing
- Ensure internet connectivity

**Memory Issues**
- Reduce chunk size for large documents
- Process papers in smaller batches
- Increase system RAM

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error messages
3. Create an issue in the repository

---

**NASA Space App Challenge Team** - Building the future of space research analysis
