"""
FastAPI-based API for the document intelligence system.
"""
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import tempfile
import os

from .document_processor import DocumentProcessor
from .text_analyzer import TextAnalyzer

app = FastAPI(
    title="AI Document Intelligence API",
    description="API for processing and analyzing documents using AI",
    version="1.0.0"
)

# Initialize processors
doc_processor = DocumentProcessor()
text_analyzer = TextAnalyzer()

class AnalysisResponse(BaseModel):
    """Response model for document analysis."""
    text: str
    analysis: Dict[str, Any]
    layout: Dict[str, Any]

from fastapi import HTTPException

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Analyze a document image and extract information.
    
    Args:
        file (UploadFile): Uploaded document image
        
    Returns:
        AnalysisResponse: Analysis results including extracted text and information
        
    Raises:
        HTTPException: If there is an error processing the document
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process document
        doc_processor.load_image(temp_path)
        doc_processor.preprocess_image()
        
        try:
            # Extract text and analyze
            text = doc_processor.extract_text()
            layout = doc_processor.detect_layout()
            analysis = text_analyzer.analyze_text(text)
        except RuntimeError as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
        
        return AnalysisResponse(
            text=text,
            analysis=analysis,
            layout=layout
        )
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

def start_server():
    """Start the FastAPI server."""
    uvicorn.run(app, host="0.0.0.0", port=8000)