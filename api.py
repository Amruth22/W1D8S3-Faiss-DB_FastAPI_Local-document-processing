from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
from pdf_rag_pipeline import PDFRAGPipeline

# Initialize FastAPI app
app = FastAPI(
    title="PDF RAG Pipeline API",
    description="API for the PDF Retrieval-Augmented Generation (RAG) Pipeline using Google's Gemini models and FAISS vector database",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    response: str
    context: List[str]
    similarity_scores: List[float]
    num_context_chunks: int

class IngestResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_embeddings: int

# Initialize PDF RAG pipeline
pdf_rag_pipeline = PDFRAGPipeline()

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check"""
    return {"message": "PDF RAG Pipeline API is running!"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "PDF RAG Pipeline API"}

@app.post("/ingest-pdf", response_model=IngestResponse, tags=["Document Processing"])
async def ingest_pdf_documents(files: List[UploadFile] = File(...)):
    """Ingest PDF documents into the RAG pipeline"""
    try:
        pdf_contents = []
        for file in files:
            if file.content_type != "application/pdf":
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            content = await file.read()
            pdf_contents.append(content)
        
        stats = pdf_rag_pipeline.ingest_pdf_documents(pdf_contents)
        return IngestResponse(
            total_documents=stats["total_documents"],
            total_chunks=stats["total_chunks"],
            total_embeddings=stats["total_embeddings"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF document ingestion failed: {str(e)}")

@app.post("/query-pdf", response_model=QueryResponse, tags=["Query"])
async def query_pdf_pipeline(request: QueryRequest):
    """Query the PDF RAG pipeline"""
    try:
        result = pdf_rag_pipeline.query(request.question, request.top_k)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return QueryResponse(
            response=result["response"],
            context=result["context"],
            similarity_scores=result["similarity_scores"],
            num_context_chunks=result["num_context_chunks"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/reset-pdf", tags=["Management"])
async def reset_pdf_pipeline():
    """Reset the PDF pipeline"""
    try:
        pdf_rag_pipeline.reset_pipeline()
        return {"message": "PDF Pipeline reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF Pipeline reset failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)