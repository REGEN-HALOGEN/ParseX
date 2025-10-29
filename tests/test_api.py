"""
Tests for the API module.
"""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import cv2
import numpy as np
from src.api import app

client = TestClient(app)

@pytest.fixture
def test_image():
    """Create a test image file."""
    # Create a white image with black text
    img = np.full((100, 300), 255, dtype=np.uint8)
    cv2.putText(img, "Test Text", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Save temporary image
    temp_path = Path("test_image.png")
    cv2.imwrite(str(temp_path), img)
    yield temp_path
    
    # Cleanup
    temp_path.unlink()

def test_analyze_endpoint(test_image):
    """Test the /analyze endpoint."""
    with open(test_image, "rb") as f:
        response = client.post("/analyze", files={"file": f})
    
    # If Tesseract is not installed, we expect a 500 error
    if response.status_code == 500:
        error_data = response.json()
        assert error_data["detail"] == "Tesseract OCR is not installed or not in PATH"
        return
    
    # If Tesseract is installed, we expect a successful response
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "text" in data
    assert "analysis" in data
    assert "layout" in data
    
    # Check analysis structure
    analysis = data["analysis"]
    assert "entities" in analysis
    assert "key_phrases" in analysis
    assert "summary" in analysis
    assert "sentiment" in analysis
    
    # Check layout structure
    layout = data["layout"]
    assert "blocks" in layout