# NASA SPACs RAG System

A Retrieval-Augmented Generation (RAG) system for NASA SPACs (Space Applications and Commercial Space) research papers. This system allows you to build a searchable knowledge base from CSV data and query it using natural language.

## Features

- **Document Processing**: Converts CSV data into searchable text chunks
- **Semantic Search**: Uses sentence transformers for semantic similarity search
- **FAISS Index**: Fast vector search using Facebook's FAISS library
- **OpenAI Integration**: Optional AI-powered answer generation
- **Flexible Data Format**: Supports various CSV column structures

## Prerequisites

- Python 3.8 or higher
- OpenAI API key (optional, for AI-generated answers)

## Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional):
   ```bash
   cp .env.template .env
   # Edit .env and add your OpenAI API key
   ```

## Data Format

Your CSV file should contain columns for different sections of research papers. The system expects these columns:

- `author`: Author name(s)
- `abstract`: Paper abstract
- `intro`: Introduction section
- `background`: Background section
- `methods`: Methods section
- `discussions`: Discussion section
- `conclusions`: Conclusions section
- `title`: Paper title (optional)

**Note**: Not all columns are required. The system will use whatever columns are available in your CSV.

## Usage

### 1. Build the Search Index

First, create a searchable index from your CSV data:

```bash
python train_rag_from_csv.py --csv your_data.csv --index_dir ./space_index --build
```

This will:
- Read your CSV file
- Convert each row into text chunks
- Generate embeddings using sentence transformers
- Build a FAISS index for fast searching
- Save everything to the specified directory

### 2. Query the System

Once you have built the index, you can query it:

```bash
python train_rag_from_csv.py --index_dir ./space_index --query "What are the main challenges in space applications?"
```

This will:
- Find the most relevant text chunks
- Display them with similarity scores
- Generate an AI answer (if OpenAI API key is configured)

### 3. Interactive Queries

You can run multiple queries by calling the script multiple times, or modify the script to add an interactive mode.

## Configuration Options

### Command Line Arguments

- `--csv`: Path to your CSV file (required for building)
- `--index_dir`: Directory to save/load the index (default: `./index_data`)
- `--build`: Build the index from CSV
- `--query`: Run a single query
- `--top_k`: Number of retrieved chunks (default: 5)
- `--openai_model`: OpenAI model to use (default: `gpt-4o-mini`)

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (optional)

## Example Workflow

1. **Prepare your data**:
   ```bash
   # Make sure your CSV has the expected columns
   head -5 your_data.csv
   ```

2. **Build the index**:
   ```bash
   python train_rag_from_csv.py --csv your_data.csv --index_dir ./space_index --build
   ```

3. **Query the system**:
   ```bash
   python train_rag_from_csv.py --index_dir ./space_index --query "What are the latest developments in satellite technology?"
   ```

## Technical Details

### Embedding Model
- Uses `all-mpnet-base-v2` sentence transformer model
- Generates 768-dimensional embeddings
- Optimized for semantic similarity search

### Chunking Strategy
- Default chunk size: 2000 characters
- Overlap: 200 characters
- Preserves document structure and context

### Search Algorithm
- Cosine similarity search using FAISS
- Normalized embeddings for optimal performance
- Configurable top-k retrieval

## Troubleshooting

### Common Issues

1. **"No module named 'sentence_transformers'"**
   - Run: `pip install sentence-transformers`

2. **"No module named 'faiss'"**
   - Run: `pip install faiss-cpu` (or `faiss-gpu` if you have CUDA)

3. **OpenAI API errors**
   - Check your API key is set correctly
   - Ensure you have sufficient API credits

4. **CSV format issues**
   - Make sure your CSV has proper headers
   - Check for encoding issues (try UTF-8)

### Performance Tips

- Use `faiss-gpu` instead of `faiss-cpu` for faster search on GPU-enabled systems
- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` based on your document types
- Use smaller `top_k` values for faster queries

## Sample Data

A sample CSV file (`sample_data.csv`) is included to help you get started. You can use it to test the system:

```bash
python train_rag_from_csv.py --csv sample_data.csv --index_dir ./test_index --build
python train_rag_from_csv.py --index_dir ./test_index --query "What is space debris?"
```

## Contributing

Feel free to modify the code to suit your specific needs:
- Adjust chunking parameters
- Change the embedding model
- Add new data processing features
- Implement additional query interfaces

## License

This project is open source. Feel free to use and modify as needed.
