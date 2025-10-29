# AI Document Intelligence Project

## Overview
This project implements an AI-powered document processing and analysis system with a user-friendly web interface.

## Features
- Document text extraction using OCR
- Named entity recognition
- Key phrase extraction
- Text summarization
- Sentiment analysis
- Document layout analysis
- Web-based user interface

## Project Structure
```
.
├── src/              # Source code
├── tests/            # Test files
├── models/           # Trained ML models
├── data/            # Data files
├── main.py          # API server
└── streamlit_app.py # Web interface
```

## Setup
1. Install Python 3.8+
2. Install Tesseract OCR:
   - Windows: Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Start the web interface:
```bash
streamlit run streamlit_app.py
```

2. Or use the API server:
```bash
python main.py
```
The API will be available at `http://localhost:8000`

## Web Interface
The web interface provides an easy way to:
- Upload documents
- View extracted text
- See named entities and key phrases
- Get text summaries
- Analyze sentiment
- Examine document layout

## API Endpoints
- POST `/analyze`: Upload and analyze a document
- GET `/docs`: API documentation (Swagger UI)

## Requirements
- Python 3.8+
- Tesseract OCR
- See requirements.txt for Python packages

## License
[License information will be added here]