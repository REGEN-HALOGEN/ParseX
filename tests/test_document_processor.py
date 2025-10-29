"""
Tests for the document processor module.
"""
import pytest
import numpy as np
import cv2
from pathlib import Path
from src.document_processor import DocumentProcessor

@pytest.fixture
def test_image():
    """Create a simple test image."""
    # Create a white image with black text
    img = np.full((100, 300), 255, dtype=np.uint8)
    cv2.putText(img, "Test Text", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save temporary image
    temp_path = Path("test_image.png")
    cv2.imwrite(str(temp_path), img)
    yield str(temp_path)
    
    # Cleanup
    temp_path.unlink()

def test_document_processor_initialization():
    """Test DocumentProcessor initialization."""
    processor = DocumentProcessor()
    assert processor.image is None
    assert processor.processed_image is None

def test_image_loading(test_image):
    """Test image loading functionality."""
    processor = DocumentProcessor()
    processor.load_image(test_image)
    assert processor.image is not None
    assert processor.processed_image is not None

def test_image_preprocessing(test_image):
    """Test image preprocessing functionality."""
    processor = DocumentProcessor()
    processor.load_image(test_image)
    processor.preprocess_image()
    assert processor.processed_image is not None
    
def test_text_extraction(test_image):
    """Test text extraction functionality."""
    processor = DocumentProcessor()
    processor.load_image(test_image)
    processor.preprocess_image()
    
    try:
        text = processor.extract_text()
        assert isinstance(text, str)
        assert len(text) > 0
    except RuntimeError as e:
        assert str(e) == "Tesseract OCR is not installed or not in PATH"

def test_layout_detection(test_image):
    """Test layout detection functionality."""
    processor = DocumentProcessor()
    processor.load_image(test_image)
    processor.preprocess_image()
    
    try:
        layout = processor.detect_layout()
        assert isinstance(layout, dict)
        assert 'blocks' in layout
    except RuntimeError as e:
        assert str(e) == "Tesseract OCR is not installed or not in PATH"