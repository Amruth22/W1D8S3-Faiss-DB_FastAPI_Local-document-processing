from typing import List, Dict, Any
from pdf_processor import PDFProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import FAISSVectorStore
from src.llm import GeminiLLM
from config.config import Config

class PDFRAGPipeline:
    """PDF-specific RAG Pipeline orchestrator"""
    
    def __init__(self):
        """Initialize all components of the PDF RAG pipeline"""
        print("Initializing PDF RAG Pipeline...")
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = FAISSVectorStore()
        self.llm = GeminiLLM()
        
        # Pipeline state
        self.is_indexed = False
        
        print("PDF RAG Pipeline initialized successfully!")
    
    def ingest_pdf_documents(self, pdf_contents: List[bytes]) -> Dict[str, Any]:
        """
        Ingest PDF documents into the RAG pipeline
        
        Args:
            pdf_contents: List of PDF file contents as bytes
            
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"Starting PDF document ingestion for {len(pdf_contents)} documents...")
        
        all_chunks = []
        total_chunks = 0
        
        # Process each PDF document
        for i, pdf_content in enumerate(pdf_contents):
            print(f"Processing PDF document {i+1}/{len(pdf_contents)}...")
            chunks = self.pdf_processor.process_pdf_content(pdf_content)
            all_chunks.extend(chunks)
            total_chunks += len(chunks)
        
        print(f"Total chunks created: {total_chunks}")
        
        # Generate embeddings for all chunks
        print("Generating embeddings...")
        embeddings = self.embedding_generator.generate_embeddings(all_chunks)
        
        # Add to vector store
        print("Adding embeddings to vector store...")
        self.vector_store.add_embeddings(embeddings, all_chunks)
        
        # Save the index
        self.vector_store.save_index("pdf_faiss_index")
        
        self.is_indexed = True
        
        stats = {
            "total_documents": len(pdf_contents),
            "total_chunks": total_chunks,
            "total_embeddings": len(embeddings),
            "vector_store_stats": self.vector_store.get_stats()
        }
        
        print("PDF document ingestion completed!")
        return stats
    
    def query(self, question: str, top_k: int = Config.TOP_K_RESULTS) -> Dict[str, Any]:
        """
        Query the PDF RAG pipeline
        
        Args:
            question: User's question
            top_k: Number of top similar chunks to retrieve
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not self.is_indexed:
            return {
                "response": "Error: No PDF documents have been indexed yet. Please ingest PDF documents first.",
                "context": [],
                "similarity_scores": [],
                "error": "No index available"
            }
        
        print(f"Processing query: {question}")
        
        # Generate embedding for the query
        print("Generating query embedding...")
        query_embedding = self.embedding_generator.generate_single_embedding(question)
        
        # Search for similar chunks
        print("Searching for relevant context...")
        similar_texts, similarity_scores = self.vector_store.search(query_embedding, top_k)
        
        if not similar_texts:
            return {
                "response": "I couldn't find any relevant information in the PDF documents to answer your question.",
                "context": [],
                "similarity_scores": [],
                "error": "No relevant context found"
            }
        
        print(f"Found {len(similar_texts)} relevant chunks")
        
        # Generate response using LLM
        print("Generating response...")
        response = self.llm.generate_response(question, similar_texts)
        
        result = {
            "response": response,
            "context": similar_texts,
            "similarity_scores": similarity_scores,
            "num_context_chunks": len(similar_texts)
        }
        
        print("Query processed successfully!")
        return result
    
    def reset_pipeline(self):
        """Reset the pipeline by clearing the vector store"""
        print("Resetting PDF pipeline...")
        self.vector_store = FAISSVectorStore()
        self.is_indexed = False
        print("PDF pipeline reset completed!")