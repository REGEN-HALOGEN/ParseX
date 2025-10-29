from setuptools import setup, find_packages

setup(
    name="doc_intelligence",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "opencv-python>=4.5.3",
        "pytesseract>=0.3.8",
        "spacy>=3.1.0",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "python-multipart>=0.0.5",
    ],
)