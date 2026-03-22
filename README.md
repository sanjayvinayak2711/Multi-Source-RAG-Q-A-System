# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system that processes PDF, DOCX, and TXT files, creates vector embeddings using sentence-transformers, and generates responses via OpenAI GPT or local context extraction.

## What's Working

### Document Processing
- **PDF Parsing**: PyPDF2 for text extraction
- **DOCX Parsing**: python-docx for Word documents
- **TXT Parsing**: UTF-8 text file reading
- **Chunking**: 300-400 word chunks with 50-word overlap, sentence boundary preservation
- **Text Cleaning**: Regex-based placeholder removal, duplicate sentence detection, basic formatting

### Vector Search
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, local inference via sentence-transformers)
- **Vector Store**: ChromaDB with cosine similarity (in-memory, no persistence configured)
- **Semantic Search**: Cosine similarity matching between query and document embeddings
- **Response Generation**: OpenAI GPT-3.5-turbo (if API key provided) or excerpt aggregation

## Architecture

```
┌──────────────┐     HTTP/JSON      ┌─────────────────────────────────────────┐
│   Browser    │◄──────────────────►│           FastAPI Backend             │
│  :3000       │                    │             :8000                       │
└──────────────┘                    │  ┌─────────────┐    ┌───────────────┐   │
                                    │  │ /api/upload │    │  /api/chat    │   │
                                    │  │             │    │               │   │
                                    │  │ PyPDF2      │    │ Query Embed   │   │
                                    │  │ python-docx │    │   (384-dim)   │   │
                                    │  │ TXT parser  │    │      │        │   │
                                    │  │     │       │    │      ▼        │   │
                                    │  │ Chunk/Embed │    │  Top-k Search │   │
                                    │  │     │       │    │  (ChromaDB)   │   │
                                    │  │     ▼       │    │      │        │   │
                                    │  │  ┌─────┐    │    │      ▼        │   │
                                    │  │  │Store│────┼────┼──►┌─────────┐   │   │
                                    │  │  └─────┘    │    │   │ Context │   │   │
                                    │  └─────────────┘    │   │   Build │   │   │
                                    │                     │   │    │    │   │   │
                                    │                     │   │    ▼    │   │   │
                                    │                     │   │  OpenAI?│   │   │
                                    │                     │   │  (GPT)  │   │   │
                                    │                     │   └────┬────┘   │   │
                                    │                     │        │        │   │
                                    │                     └────────┼────────┘   │
                                    │                              │            │
                                    └──────────────────────────────┼────────────┘
                                                                   │
                    ┌─────────────────┐     ┌───────────────────┐    │
                    │  Sentence-      │◄────│     CHROMADB      │    │
                    │  Transformers   │     │   (in-memory)     │    │
                    │  (all-MiniLM)   │     │  • 384-dim vectors│    │
                    └─────────────────┘     │  • Cosine sim     │◄───┘
                                            │  • Metadata       │
                                            └───────────────────┘
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
First run downloads embedding model (~80MB).

### 2. Optional: OpenAI API Key
```bash
set OPENAI_API_KEY=your_key_here  # Windows
export OPENAI_API_KEY=your_key_here  # Linux/Mac
```
Without this key, the system returns cleaned document excerpts instead of GPT-generated responses.

### 3. Run System
```bash
run.bat  # Windows only
```

### 4. Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## How It Works

1. **Upload Documents** → Files parsed, cleaned, chunked (~350 words each)
2. **Create Embeddings** → Sentence-transformers encodes chunks to 384-dim vectors
3. **Store in ChromaDB** → Vectors indexed with cosine similarity metric
4. **Query Processing** → User query embedded, top-k chunks retrieved via similarity search
5. **Response Generation** → OpenAI GPT-3.5 generates answer from retrieved chunks (or raw excerpts if no API key)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload PDF/DOCX/TXT files |
| `/api/documents` | GET | List uploaded documents |
| `/api/chat` | POST | Query documents, get response |
| `/api/stats` | GET | System statistics |
| `/api/clear` | POST | Clear all documents and history |

## Operating Modes

### With OpenAI API Key
- GPT-3.5-turbo generates natural language responses
- Requires valid API key with available credits
- Response time depends on OpenAI API latency (~500-2000ms)

### Without API Key (Local Mode)
- Returns cleaned excerpts from retrieved chunks
- Zero API costs
- Faster responses (~100-300ms)

## Technical Details

### Request/Response Schemas (Pydantic)
- `ChatRequest`: `{message: str}`
- `ChatResponse`: `{message: str, sources: List[str], timestamp: datetime}`
- `DocumentInfo`: `{id, name, type, size, status, upload_time}`

### Performance Characteristics
- Embedding generation: ~50-100ms per 350-word chunk (CPU)
- Vector search: ~10-30ms for 1000+ chunks
- Memory usage: ~200MB base + model (~80MB) + document storage
- No database: Uses in-memory Python lists for metadata, ChromaDB for vectors

## Current Limitations

- **No persistent database**: Document metadata stored in memory (lost on restart)
- **No user authentication**: Single-user system
- **No file storage**: Temporary file handling only
- **CPU-only embeddings**: GPU acceleration not configured
- **ChromaDB in-memory**: Vector persistence depends on ChromaDB configuration
- **Basic retrieval only**: No hybrid search, re-ranking, or query expansion

## Project Assessment

**Honest evaluation for recruiters:**

This is a **tutorial-level RAG implementation** demonstrating core concepts:
- Vector embeddings and similarity search
- Document chunking strategies
- FastAPI backend architecture
- Basic LLM integration

**What this project is NOT:**
- Production-ready system (no auth, no persistence, no scaling)
- Novel research contribution
- Advanced RAG (no agents, reranking, query rewriting, or evaluation pipeline)

**Why it exists:** Learning foundational RAG patterns before building differentiated features.

## Differentiation Roadmap

To make this recruiter-impactful, implement:

1. **Hybrid Search** (BM25 + Vector) - Shows understanding beyond "similarity = search"
2. **Cross-Encoder Re-ranking** - Re-score top-k retrieved chunks before LLM context
3. **Query Rewriting** - LLM-expands queries to improve recall
4. **Evaluation Pipeline** - RAGAS metrics (faithfulness, answer relevance) on test set
5. **Async Processing** - Celery + Redis for non-blocking document ingestion

## Roadmap

Potential enhancements:
- PostgreSQL + pgvector for persistent storage
- JWT-based user authentication
- Redis for query caching
- S3/local file system for document storage
- Streaming responses

## Requirements

- Python 3.8+
- 4GB RAM minimum
- ~100MB disk space + documents + model
- GPU optional (CPU-optimized)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named X" | `pip install -r requirements.txt` |
| "Backend not responding" | Check port 8000 is free |
| "No text extracted" | PDF may be image-based (needs OCR) |
| "OpenAI API error" | Verify API key validity and credits |

## License

MIT License
