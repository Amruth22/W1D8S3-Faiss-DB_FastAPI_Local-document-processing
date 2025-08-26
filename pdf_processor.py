import fitz  # PyMuPDF
import io
from typing import List
from src.document_processor import DocumentProcessor

class PDFProcessor:
    """Handles PDF document processing and text extraction"""
    
    def __init__(self):
        """Initialize PDF processor"""
        self.document_processor = DocumentProcessor()
    
    def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted text from PDF
        """
        try:
            # Create a BytesIO object from the PDF content
            pdf_stream = io.BytesIO(pdf_content)
            
            # Open the PDF with PyMuPDF
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            # Extract text from all pages
            text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text() + "\n"
            
            doc.close()
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def process_pdf_content(self, pdf_content: bytes) -> List[str]:
        """
        Process PDF content into chunks
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            List of processed text chunks
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_content)
        
        # Process text using document processor
        chunks = self.document_processor.process_document(text)
        
        return chunks