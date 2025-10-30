"""
Data collection script for training custom extraction models.
Uses existing OCR capabilities to process documents and generate training data.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import tempfile
from PIL import Image

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.document_processor import DocumentProcessor


class TrainingDataCollector:
    """Collects training data by processing documents with OCR."""

    def __init__(self, output_dir: str = "backend/training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.doc_processor = DocumentProcessor()

    def process_document_batch(self, document_paths: List[str], batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process a batch of documents and extract OCR data.

        Args:
            document_paths: List of paths to document images
            batch_size: Number of documents to process in each batch

        Returns:
            List of processed document data
        """
        processed_data = []

        for i in range(0, len(document_paths), batch_size):
            batch = document_paths[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} with {len(batch)} documents...")

            for doc_path in batch:
                try:
                    doc_data = self._process_single_document(doc_path)
                    if doc_data:
                        processed_data.append(doc_data)
                        print(f"✓ Processed: {os.path.basename(doc_path)}")
                    else:
                        print(f"✗ Failed to process: {os.path.basename(doc_path)}")
                except Exception as e:
                    print(f"✗ Error processing {os.path.basename(doc_path)}: {str(e)}")

        return processed_data

    def _process_single_document(self, doc_path: str) -> Dict[str, Any] | None:
        """Process a single document and extract OCR data."""
        try:
            # Load and preprocess image
            self.doc_processor.load_image(doc_path)
            self.doc_processor.preprocess_image()

            # Extract text
            text = self.doc_processor.extract_text()
            if not text or not text.strip():
                return None

            # Detect layout
            layout = self.doc_processor.detect_layout()

            # Create document data structure
            doc_data = {
                "document_path": doc_path,
                "filename": os.path.basename(doc_path),
                "extracted_text": text,
                "layout": layout,
                "metadata": {
                    "image_size": self._get_image_size(doc_path),
                    "processing_timestamp": None,  # Will be set when saving
                },
                "ground_truth": None  # To be filled by labeling process
            }

            return doc_data

        except Exception as e:
            print(f"Error processing {doc_path}: {str(e)}")
            return None

    def _get_image_size(self, image_path: str) -> Dict[str, int]:
        """Get image dimensions."""
        try:
            with Image.open(image_path) as img:
                return {"width": img.width, "height": img.height}
        except Exception:
            return {"width": 0, "height": 0}

    def save_processed_data(self, processed_data: List[Dict[str, Any]], output_file: str | None = None) -> str:
        """
        Save processed data to JSON file.

        Args:
            processed_data: List of processed document data
            output_file: Output filename (optional)

        Returns:
            Path to saved file
        """
        if not output_file:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"training_data_{timestamp}.json"

        output_path = self.output_dir / output_file

        # Add processing timestamps
        import datetime
        timestamp = datetime.datetime.now().isoformat()
        for doc in processed_data:
            doc["metadata"]["processing_timestamp"] = timestamp

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(processed_data)} documents to {output_path}")
        return str(output_path)

    def collect_from_directory(self, input_dir: str, extensions: List[str] | None = None) -> List[str]:
        """
        Collect all document paths from a directory.

        Args:
            input_dir: Input directory path
            extensions: List of file extensions to include (default: common image formats)

        Returns:
            List of document file paths
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']

        input_path = Path(input_dir)
        document_paths = []

        for ext in extensions:
            document_paths.extend(input_path.glob(f"**/*{ext}"))
            document_paths.extend(input_path.glob(f"**/*{ext.upper()}"))

        return [str(path) for path in sorted(document_paths)]


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Collect training data for custom extraction models")
    parser.add_argument("--input-dir", required=True, help="Directory containing documents to process")
    parser.add_argument("--output-dir", default="backend/training_data", help="Output directory for processed data")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of documents to process per batch")
    parser.add_argument("--output-file", help="Output filename (optional)")
    parser.add_argument("--extensions", nargs='+', default=['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
                       help="File extensions to process")

    args = parser.parse_args()

    # Initialize collector
    collector = TrainingDataCollector(args.output_dir)

    # Collect document paths
    print(f"Scanning {args.input_dir} for documents...")
    document_paths = collector.collect_from_directory(args.input_dir, args.extensions)
    print(f"Found {len(document_paths)} documents")

    if not document_paths:
        print("No documents found. Exiting.")
        return

    # Process documents
    processed_data = collector.process_document_batch(document_paths, args.batch_size)

    # Save results
    if processed_data:
        output_path = collector.save_processed_data(processed_data, args.output_file)
        print(f"Training data collection complete. Processed {len(processed_data)} documents.")
        print(f"Output saved to: {output_path}")
    else:
        print("No documents were successfully processed.")


if __name__ == "__main__":
    main()
