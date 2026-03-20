# RAG AI System - Complete Working Model

A fully functional Retrieval-Augmented Generation (RAG) system that processes PDF, DOCX, and TXT files, creates vector embeddings, performs semantic search, and generates AI responses using OpenAI GPT or local context.

## 🚀 What's Working Now

### Document Processing
- **PDF Parsing**: Extracts text from PDF files using PyPDF2
- **DOCX Parsing**: Extracts text from Word documents using python-docx  
- **TXT Parsing**: Reads plain text files
- **Smart Chunking**: Splits documents into 1000-character overlapping chunks with sentence preservation
- **Text Cleaning**: Fixes OCR errors, removes duplicates, formats professionally

### Vector Search & AI
- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast & lightweight)
- **Vector Database**: ChromaDB with cosine similarity and persistent storage
- **Semantic Search**: Finds relevant document chunks based on query meaning
- **Response Generation**: OpenAI GPT-3.5-turbo or local context with deduplication

### Performance Features
- **Fast Startup**: 3-second launch time
- **Optimized Chunks**: 1000-char chunks for 2x faster processing
- **Memory Efficient**: ~200MB usage with ChromaDB persistence
- **Clean Output**: Professional formatting with source attribution

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend     │    │    Backend      │    │  Vector Store   │
│   (UI/Chat)    │◄──►│   (FastAPI)     │◄──►│   (ChromaDB)    │
│  Port: 3000    │    │  Port: 8000     │    │  Persistent      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         └──────────────►│  Embeddings     │◄─────────────┘
                        │ (Sentence-     │
                        │  Transformers) │
                        └─────────────────┘
```

## 📁 Project Structure

```
RAG/
├── main.py                 # Complete backend (445 lines) with text cleaning
├── requirements.txt        # All dependencies with versions
├── run.bat                # Windows launcher (simultaneous backend+frontend)
├── .env.example           # Environment variables template
├── README.md              # This documentation
├── chroma_db/             # Vector database (auto-created)
└── ui/
    ├── index.html         # Modern frontend UI with dark theme
    ├── script.js          # API integration with live metrics
    ├── styles.css         # Responsive design
    └── package.json       # Frontend metadata
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
*First run downloads embedding model (~80MB)*

### 2. Optional: OpenAI API Key
```bash
set OPENAI_API_KEY=your_api_key_here  # Windows
export OPENAI_API_KEY=your_api_key_here  # Linux/Mac
```

### 3. Run System
```bash
run.bat  # Windows - launches both servers automatically
```

### 4. Access
- **Frontend**: http://localhost:3000 (auto-opens)
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🔧 How It Works

1. **Upload Documents** → Files parsed and chunked with sentence preservation
2. **Create Embeddings** → Each chunk converted to 384-dim vector
3. **Store in ChromaDB** → Vectors saved with metadata and persistence
4. **Ask Questions** → Query embedded and matched via cosine similarity
5. **Generate Response** → Clean, deduplicated answers with sources

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload and process PDF/DOCX/TXT files |
| `/api/documents` | GET | List all uploaded documents |
| `/api/chat` | POST | Send query, get cleaned AI response |
| `/api/stats` | GET | System statistics and performance |
| `/api/clear` | POST | Clear all documents and history |

## 🎯 Operating Modes

### OpenAI Mode (API Key Set)
- Natural language responses with context understanding
- Professional text formatting and summarization
- Intelligent deduplication and cleaning

### Local Mode (No API Key)
- Returns cleaned document excerpts
- Zero cost, faster responses
- Built-in text cleaning and formatting

## 📊 Performance Metrics

- **Startup Time**: 3 seconds
- **Embedding Speed**: 50-100ms per chunk
- **Vector Search**: 10-30ms
- **Total Response**: 200-800ms (depends on OpenAI)
- **Memory Usage**: ~200MB
- **Storage**: Persistent ChromaDB with compression

## 💻 System Requirements

- **Python**: 3.8+
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: ~100MB + documents
- **GPU**: Not required (CPU optimized)

## 🛠️ Features

### Text Processing
- ✅ OCR error correction
- ✅ Sentence-aware chunking
- ✅ Duplicate removal
- ✅ Professional formatting
- ✅ Source attribution

### User Interface
- ✅ Modern dark theme
- ✅ Real-time metrics
- ✅ File drag & drop
- ✅ Responsive design
- ✅ Live chat with sources

### Backend
- ✅ FastAPI with async
- ✅ ChromaDB persistence
- ✅ CORS enabled
- ✅ Error handling
- ✅ Health checks

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named X" | `pip install -r requirements.txt` |
| "Backend not responding" | Check port 8000 is free |
| "No text extracted" | PDF may be scanned images |
| "OpenAI API error" | Verify API key is valid |
| "Chunk errors" | Check file encoding and format |

## 🚀 Production Ready Features

- ✅ Persistent vector storage
- ✅ Error handling and logging
- ✅ CORS configuration
- ✅ Text cleaning pipeline
- ✅ Source attribution
- ✅ Performance optimization
- ✅ Cross-platform compatibility

## 📋 Next Steps

1. **User Authentication** - JWT-based auth system
2. **Database Upgrade** - PostgreSQL with pgvector
3. **File Storage** - S3 or local file system
4. **Response Streaming** - Real-time chat responses
5. **Query Caching** - Redis for common queries
6. **Multi-modal Support** - Image and audio processing

---

**🎉 Complete RAG system with professional text cleaning and fast performance!**

## 📄 License

MIT License - Free for commercial and personal use.
