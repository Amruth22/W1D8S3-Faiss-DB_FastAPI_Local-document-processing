# PDF RAG Pipeline API

A FastAPI-based PDF document processing and Retrieval-Augmented Generation (RAG) system using Google Gemini models and FAISS vector database.

## 🏗️ Architecture

**Technology Stack:**
- **API Framework**: FastAPI with automatic OpenAPI documentation
- **PDF Processing**: PyMuPDF for text extraction
- **Vector Database**: FAISS for similarity search
- **LLM**: Google Gemini 2.5 Flash for response generation
- **Embeddings**: Google Gemini Embedding Model (3072 dimensions)
- **Server**: Uvicorn ASGI server

**System Flow:**
```
PDF Upload → PDF Processor → Document Processor → Embeddings → FAISS Vector Store
                                                                      ↓
User API Query → Embedding Generation → Vector Search → Context Retrieval → LLM Response
```

## 📁 Project Structure

```
W1D8S3-Faiss-DB_FastAPI_Local-document-processing/
├── main.py                    # FastAPI server startup script
├── api.py                     # FastAPI endpoints and Pydantic models
├── pdf_rag_pipeline.py        # PDF-specific RAG orchestrator
├── pdf_processor.py           # PDF text extraction using PyMuPDF
├── requirements.txt           # Python dependencies
├── .env.template             # Environment variables template
├── config/
│   ├── __init__.py
│   └── config.py             # Configuration management
├── src/
│   ├── __init__.py
│   ├── embeddings.py         # Gemini embedding generation
│   ├── vector_store.py       # FAISS vector database operations
│   ├── llm.py               # Gemini LLM integration
│   └── document_processor.py # Text chunking and processing
├── test_imports.py           # Import testing script
└── test_pymupdf.py          # PyMuPDF testing script
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables
```bash
# Copy the template file
cp .env.template .env

# Edit .env and add your Google API key
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: [Google AI Studio](https://aistudio.google.com/)

### 3. Test Installation
```bash
python test_imports.py
```

### 4. Start the API Server
```bash
# Method 1: Using main.py (recommended)
python main.py

# Method 2: With custom options
python main.py --host 0.0.0.0 --port 8000 --reload

# Method 3: Direct uvicorn
uvicorn api:app --host 127.0.0.1 --port 8080 --reload
```

### 5. Access API Documentation
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc
- **OpenAPI JSON**: http://localhost:8080/openapi.json

## 🔌 API Endpoints

### Health Check
```bash
# Check if API is running
curl http://localhost:8080/health
```

### Upload PDF Documents
```bash
curl -X POST "http://localhost:8080/ingest-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response:**
```json
{
  "total_documents": 2,
  "total_chunks": 45,
  "total_embeddings": 45
}
```

### Query Documents
```bash
curl -X POST "http://localhost:8080/query-pdf" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "response": "Based on the provided context...",
  "context": ["Relevant text chunk 1", "Relevant text chunk 2"],
  "similarity_scores": [0.92, 0.87],
  "num_context_chunks": 2
}
```

### Reset Pipeline
```bash
curl -X POST "http://localhost:8080/reset-pdf"
```

## 🐍 Python API Usage

```python
import requests

# Upload PDF documents
with open('document.pdf', 'rb') as file:
    files = {'files': file}
    response = requests.post('http://localhost:8080/ingest-pdf', files=files)
    print(response.json())

# Query the system
query_data = {
    "question": "What are the key findings?",
    "top_k": 3
}
response = requests.post('http://localhost:8080/query-pdf', json=query_data)
print(response.json())
```

## 📋 API Reference

### Request Models

**QueryRequest:**
```json
{
  "question": "string",
  "top_k": 5  // optional, default: 5
}
```

### Response Models

**QueryResponse:**
```json
{
  "response": "string",
  "context": ["string"],
  "similarity_scores": [0.0],
  "num_context_chunks": 0
}
```

**IngestResponse:**
```json
{
  "total_documents": 0,
  "total_chunks": 0,
  "total_embeddings": 0
}
```

## ⚙️ Configuration

Edit `config/config.py` to customize:

```python
class Config:
    # Model configurations
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "gemini-embedding-001"
    
    # RAG configurations
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 5
    
    # FAISS configurations
    VECTOR_DIMENSION = 3072  # Gemini embedding dimension
    FAISS_INDEX_PATH = "pdf_faiss_index"
```

## 🔧 Development

### Command Line Options
```bash
python main.py --help

# Options:
#   --host HOST     Host for the FastAPI server (default: 127.0.0.1)
#   --port PORT     Port for the FastAPI server (default: 8080)
#   --reload        Enable auto-reload for development
```

### Testing
```bash
# Test all imports
python test_imports.py

# Test PyMuPDF specifically
python test_pymupdf.py
```

## 🌟 Features

- ✅ **FastAPI Integration** - Modern, fast web framework
- ✅ **PDF Processing** - Extract text from PDF documents
- ✅ **Vector Search** - FAISS-powered similarity search
- ✅ **AI Responses** - Google Gemini 2.5 Flash integration
- ✅ **Automatic Documentation** - OpenAPI/Swagger UI
- ✅ **CORS Support** - Cross-origin requests enabled
- ✅ **File Validation** - PDF-only uploads
- ✅ **Error Handling** - Comprehensive error responses
- ✅ **Persistent Storage** - Automatic index saving/loading

## 🐛 Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Ensure you're in the correct directory
cd W1D8S3-Faiss-DB_FastAPI_Local-document-processing
python test_imports.py
```

**API Key Issues:**
```bash
# Check if .env file exists and has the correct key
cat .env
# Should show: GOOGLE_API_KEY=your_actual_key
```

**Port Already in Use:**
```bash
# Use a different port
python main.py --port 8081
```

**PDF Upload Fails:**
```bash
# Ensure file is actually a PDF
file document.pdf
# Should show: PDF document
```

## 📚 Dependencies

Core dependencies from `requirements.txt`:
- `google-genai>=1.29.0` - Google Gemini API
- `faiss-cpu>=1.7.4` - Vector similarity search
- `fastapi>=0.100.0` - Web framework
- `uvicorn>=0.21.0` - ASGI server
- `PyMuPDF>=1.26.0` - PDF processing
- `python-dotenv==1.0.0` - Environment variables
- `pydantic>=2.0.0` - Data validation

## 🚀 Production Deployment

For production use, consider:
1. **Environment Variables**: Use secure environment variable management
2. **HTTPS**: Enable SSL/TLS encryption
3. **Rate Limiting**: Add rate limiting middleware
4. **Authentication**: Implement API key authentication
5. **Monitoring**: Add logging and health checks
6. **Scaling**: Use multiple workers with Gunicorn
7. **Database**: Consider external vector database for large-scale deployment

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ using FastAPI, Google Gemini, and FAISS**