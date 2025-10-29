# AI Document Intelligence Project

## Overview
This project implements an AI-powered document processing and analysis system with a user-friendly web interface built using Streamlit and a robust API server powered by FastAPI.

## Features
- Document text extraction using OCR (Tesseract)
- Named entity recognition using spaCy and transformers
- Key phrase extraction with scikit-learn and transformers
- Text summarization powered by Hugging Face transformers
- Sentiment analysis using torch-based models
- Document layout analysis with OpenCV
- Web-based user interface built with Streamlit
- RESTful API with FastAPI for programmatic access

## Frameworks and Libraries Used
- **Streamlit**: For the interactive web interface (ParseX.py)
- **FastAPI**: For the API server (main.py) with automatic documentation
- **Transformers & Torch**: For advanced NLP tasks like summarization and entity recognition
- **spaCy**: For natural language processing and named entity recognition
- **OpenCV**: For image preprocessing and computer vision tasks
- **Tesseract (pytesseract)**: For optical character recognition (OCR)
- **NumPy & Pandas**: For data manipulation and analysis
- **scikit-learn**: For machine learning utilities
- **pdf2image & python-docx**: For handling PDF and DOCX document formats

## Project Structure
```
.
├── src/                    # Source code
│   ├── __init__.py
│   ├── api.py              # API endpoints
│   ├── document_processor.py # OCR and image processing
│   └── text_analyzer.py    # NLP analysis
├── tests/                  # Test files
├── models/                 # Trained ML models
├── data/                   # Data files
├── main.py                 # FastAPI server
├── ParseX.py               # Streamlit web interface
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── README.md               # This file
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
4. Download the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

## Usage
1. Start the web interface:
```bash
streamlit run ParseX.py
```

2. Or use the API server:
```bash
python main.py
```
The API will be available at `http://localhost:8000`

## Web Interface
The Streamlit web interface provides an easy way to:
- Upload documents (PNG, JPG, JPEG, PDF, DOCX)
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
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
