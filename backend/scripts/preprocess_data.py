"""
Data preprocessing script for preparing training data for custom extraction models.
Handles data normalization, format conversion, and feature engineering.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re
import pandas as pd
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Preprocesses training data for ML model training."""

    def __init__(self, data_dir: str = "backend/training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_labeled_data(self, labeled_file: str) -> List[Dict[str, Any]]:
        """Load labeled training data."""
        labeled_path = Path(labeled_file)
        with open(labeled_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def normalize_text(self, text: str) -> str:
        """Normalize extracted text for consistent processing."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep numbers and basic punctuation
        text = re.sub(r'[^\w\s\.,\-/\$₹€£%]', '', text)

        # Normalize currency symbols
        text = re.sub(r'[₹€£]', '$', text)  # Convert to dollar for consistency

        return text.strip()

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text for classical ML models."""
        features = {}

        # Basic text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['line_count'] = len(text.split('\n'))

        # Pattern-based features
        features['has_invoice'] = 1 if re.search(r'\binvoice\b', text, re.IGNORECASE) else 0
        features['has_receipt'] = 1 if re.search(r'\breceipt\b', text, re.IGNORECASE) else 0
        features['has_vendor'] = 1 if re.search(r'\bvendor\b|\bsupplier\b|\bfrom\b', text, re.IGNORECASE) else 0
        features['has_gst'] = 1 if re.search(r'\bgst\b|\btin\b|\btax\b', text, re.IGNORECASE) else 0
        features['has_amount'] = 1 if re.search(r'\$\s*\d+|\d+\.\d+', text) else 0
        features['has_date'] = 1 if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text) else 0

        # Currency amounts found
        amounts = re.findall(r'\$\s*([\d,]+\.?\d*)', text)
        features['num_amounts'] = len(amounts)

        return features

    def create_training_examples(self, labeled_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Create training examples for NER and classification.

        Args:
            labeled_data: Labeled training data

        Returns:
            Tuple of (ner_examples, classification_examples)
        """
        ner_examples = []
        classification_examples = []

        for doc in labeled_data:
            # Normalize text
            normalized_text = self.normalize_text(doc['original_text'])

            # Extract features for classification
            features = self.extract_features(normalized_text)

            # Add document type if available
            ground_truth = doc.get('ground_truth', {})
            doc_type = ground_truth.get('document_type', 'unknown')

            # Classification example
            clf_example = features.copy()
            clf_example['document_type'] = doc_type
            classification_examples.append(clf_example)

            # NER example
            ner_example = {
                "text": normalized_text,
                "entities": []
            }

            # Convert ground truth to NER format
            for field, value in ground_truth.items():
                if isinstance(value, str) and value:
                    # Find value in normalized text
                    start = normalized_text.find(value.lower())
                    if start != -1:
                        end = start + len(value)
                        ner_example["entities"].append({
                            "start": start,
                            "end": end,
                            "label": field.upper(),
                            "text": value
                        })

            ner_examples.append(ner_example)

        return ner_examples, classification_examples

    def save_datasets(self, examples: List[Dict[str, Any]], data_type: str) -> Dict[str, str]:
        """
        Save examples to train/val/test splits.

        Args:
            examples: Training examples
            data_type: Type of data ("ner" or "classification")

        Returns:
            Dictionary with file paths
        """
        if len(examples) < 3:
            # If too few examples, save all as training
            train_file = self.data_dir / f"{data_type}_train.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')

            return {"train": str(train_file)}

        # Split into train/val/test
        train_data, temp_data = train_test_split(examples, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        files = {}
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            file_path = self.data_dir / f"{data_type}_{split_name}.jsonl"
            with open(file_path, 'w', encoding='utf-8') as f:
                for example in split_data:
                    f.write(json.dumps(example, ensure_ascii=False) + '\n')
            files[split_name] = str(file_path)

        return files

    def convert_to_csv(self, classification_examples: List[Dict[str, Any]], output_file: str) -> str:
        """Convert classification examples to CSV format."""
        df = pd.DataFrame(classification_examples)
        output_path = self.data_dir / output_file
        df.to_csv(output_path, index=False)
        return str(output_path)

    def process_pipeline(self, labeled_file: str) -> Dict[str, Any]:
        """
        Run the complete preprocessing pipeline.

        Args:
            labeled_file: Path to labeled data file

        Returns:
            Dictionary with paths to all generated files
        """
        print("Loading labeled data...")
        labeled_data = self.load_labeled_data(labeled_file)

        print("Creating training examples...")
        ner_examples, classification_examples = self.create_training_examples(labeled_data)

        print(f"Generated {len(ner_examples)} NER examples and {len(classification_examples)} classification examples")

        # Save NER datasets
        ner_files = self.save_datasets(ner_examples, "ner")

        # Save classification datasets
        clf_files = self.save_datasets(classification_examples, "classification")

        # Convert classification to CSV
        csv_file = self.convert_to_csv(classification_examples, "classification_full.csv")

        return {
            "ner": ner_files,
            "classification": clf_files,
            "csv": csv_file,
            "stats": {
                "total_documents": len(labeled_data),
                "ner_examples": len(ner_examples),
                "classification_examples": len(classification_examples)
            }
        }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Preprocess training data for custom extraction models")
    parser.add_argument("--labeled-file", required=True, help="Path to labeled training data file")
    parser.add_argument("--output-dir", default="backend/training_data", help="Output directory")

    args = parser.parse_args()

    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.output_dir)

    # Run preprocessing pipeline
    results = preprocessor.process_pipeline(args.labeled_file)

    print("\nPreprocessing complete!")
    print("Generated files:")
    for category, files in results.items():
        if category != "stats":
            print(f"  {category.upper()}:")
            if isinstance(files, dict):
                for split, path in files.items():
                    print(f"    {split}: {path}")
            else:
                print(f"    {files}")

    print(f"\nStatistics: {results['stats']}")


if __name__ == "__main__":
    main()
