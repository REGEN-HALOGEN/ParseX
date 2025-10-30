"""
Manual labeling interface for creating ground truth data for training.
Provides a simple script to label extracted text with invoice/receipt fields.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import re


class DataLabeler:
    """Interactive labeling interface for training data."""

    def __init__(self, data_file: str):
        self.data_file = Path(data_file)
        self.data = self._load_data()
        self.current_index = 0

        # Define labeling schema for invoices/receipts
        self.label_schema = {
            "document_type": ["invoice", "receipt", "bill", "statement"],
            "vendor_name": "string",
            "vendor_address": "string",
            "vendor_gst": "string",
            "invoice_number": "string",
            "invoice_date": "date",
            "due_date": "date",
            "line_items": [
                {
                    "description": "string",
                    "quantity": "number",
                    "unit_price": "number",
                    "total": "number"
                }
            ],
            "subtotal": "number",
            "tax_amount": "number",
            "total_amount": "number",
            "payment_terms": "string"
        }

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load processed training data."""
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_file}")

        with open(self.data_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _extract_potential_fields(self, text: str) -> Dict[str, List[str]]:
        """Extract potential field values using regex patterns."""
        patterns = {
            "invoice_number": [
                r"invoice\s*#?\s*:?\s*([A-Z0-9\-]+)",
                r"inv\s*#?\s*:?\s*([A-Z0-9\-]+)",
                r"bill\s*#?\s*:?\s*([A-Z0-9\-]+)"
            ],
            "gst": [
                r"gst\s*#?\s*:?\s*([A-Z0-9]+)",
                r"tin\s*#?\s*:?\s*([A-Z0-9]+)",
                r"vat\s*#?\s*:?\s*([A-Z0-9]+)"
            ],
            "dates": [
                r"date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
            ],
            "amounts": [
                r"total\s*:?\s*[\$₹€£]?\s*([\d,]+\.?\d*)",
                r"amount\s*:?\s*[\$₹€£]?\s*([\d,]+\.?\d*)"
            ]
        }

        results = {}
        for field, pattern_list in patterns.items():
            results[field] = []
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                results[field].extend(matches)

        return results

    def label_all_documents(self, output_file: str = None) -> str:
        """
        Label all documents in the dataset.

        Args:
            output_file: Path to save labeled data

        Returns:
            Path to labeled data file
        """
        if output_file is None:
            output_file = self.data_file.parent / "labeled_data.json"

        labeled_data = []

        for doc in self.data:
            print(f"\nProcessing document {len(labeled_data) + 1}/{len(self.data)}")
            print(f"Text: {doc['extracted_text'][:200]}...")

            # Extract potential fields
            potential_fields = self._extract_potential_fields(doc['extracted_text'])

            # Create ground truth (simplified - in practice this would be interactive)
            ground_truth = {}

            # Simple rule-based labeling for demo
            text = doc['extracted_text'].lower()

            if 'invoice' in text:
                ground_truth['document_type'] = 'invoice'
            elif 'receipt' in text:
                ground_truth['document_type'] = 'receipt'

            # Extract invoice number
            if potential_fields['invoice_number']:
                ground_truth['invoice_number'] = potential_fields['invoice_number'][0]

            # Extract GST
            if potential_fields['gst']:
                ground_truth['vendor_gst'] = potential_fields['gst'][0]

            # Extract amounts
            if potential_fields['amounts']:
                ground_truth['total_amount'] = float(potential_fields['amounts'][0].replace(',', ''))

            labeled_data.append({
                "original_text": doc['extracted_text'],
                "ground_truth": ground_truth,
                "potential_fields": potential_fields
            })

        # Save labeled data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labeled_data, f, indent=2, ensure_ascii=False)

        print(f"Labeled data saved to {output_file}")
        return str(output_file)

    def generate_training_format(self, labeled_file: str, output_file: str = None) -> str:
        """
        Convert labeled data to training format for NER models.

        Args:
            labeled_file: Path to labeled data
            output_file: Output file for training data

        Returns:
            Path to training data file
        """
        if output_file is None:
            output_file = Path(labeled_file).parent / "training_data.jsonl"

        with open(labeled_file, 'r', encoding='utf-8') as f:
            labeled_data = json.load(f)

        training_examples = []

        for doc in labeled_data:
            ground_truth = doc.get('ground_truth')
            if not ground_truth:
                continue

            # Create training example
            example = {
                "text": doc['original_text'],
                "entities": []
            }

            # Convert ground truth to entity format
            # This is a simplified conversion - in practice, you'd need proper NER annotation
            text = doc['original_text']

            # Simple entity extraction based on labeled fields
            for field, value in ground_truth.items():
                if isinstance(value, str) and value:
                    # Find the value in text and create entity annotation
                    start = text.find(value)
                    if start != -1:
                        end = start + len(value)
                        example["entities"].append({
                            "start": start,
                            "end": end,
                            "label": field.upper(),
                            "text": value
                        })

            if example["entities"]:
                training_examples.append(example)

        # Save in JSONL format for training
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in training_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"Generated {len(training_examples)} training examples in {output_file}")
        return str(output_file)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Label training data for custom extraction models")
    parser.add_argument("--data-file", required=True, help="Path to processed training data file")
    parser.add_argument("--output-file", help="Output file for labeled data")
    parser.add_argument("--generate-training", action="store_true",
                       help="Generate training format from labeled data")

    args = parser.parse_args()

    if args.generate_training:
        # Generate training format
        labeler = DataLabeler(args.data_file)
        training_file = labeler.generate_training_format(args.data_file, args.output_file)
        print(f"Training data generated: {training_file}")
    else:
        # Interactive labeling
        labeler = DataLabeler(args.data_file)
        labeler.label_all_documents(args.output_file)


if __name__ == "__main__":
    main()
