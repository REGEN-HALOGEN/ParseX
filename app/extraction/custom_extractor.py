"""
Custom extractor module for loading and using trained models for document information extraction.
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import joblib
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import openai
import re


class CustomExtractor:
    """Main class for custom document information extraction."""

    def __init__(self, model_dir: str = "backend/models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.tokenizers = {}
        self.encoders = {}

        # Initialize OpenAI client if available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()

        # Load available models
        self._load_models()

    def _load_models(self):
        """Load all available trained models."""
        if not self.model_dir.exists():
            print(f"Model directory {self.model_dir} does not exist")
            return

        # Load classical ML models
        self._load_classical_models()

        # Load LLM models
        self._load_llm_models()

        print(f"Loaded models: {list(self.models.keys())}")

    def _load_classical_models(self):
        """Load classical ML models (Random Forest, CRF)."""
        # Load Random Forest models
        rf_files = list(self.model_dir.glob("rf_*.pkl"))
        for rf_file in rf_files:
            model_name = rf_file.stem
            try:
                self.models[model_name] = joblib.load(rf_file)

                # Load corresponding encoder
                encoder_file = rf_file.parent / f"{model_name}_encoder.pkl"
                if encoder_file.exists():
                    self.encoders[model_name] = joblib.load(encoder_file)

                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

        # Load CRF model
        crf_file = self.model_dir / "crf_ner.pkl"
        if crf_file.exists():
            try:
                self.models["crf_ner"] = joblib.load(crf_file)
                print("Loaded crf_ner")
            except Exception as e:
                print(f"Failed to load crf_ner: {e}")

    def _load_llm_models(self):
        """Load LLM models (HuggingFace and OpenAI)."""
        # Load HuggingFace models
        hf_files = list(self.model_dir.glob("hf_*.pkl"))
        for hf_file in hf_files:
            model_name = hf_file.stem
            try:
                self.models[model_name] = joblib.load(hf_file)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

        # Load OpenAI fine-tuned models
        openai_files = list(self.model_dir.glob("openai_*.json"))
        for openai_file in openai_files:
            model_name = openai_file.stem
            try:
                with open(openai_file, 'r') as f:
                    model_info = json.load(f)
                self.models[model_name] = model_info
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")

    def extract_document_info(self, text: str, model_type: str = "auto") -> Dict[str, Any]:
        """
        Extract document information using specified model type.

        Args:
            text: Document text to extract from
            model_type: "auto", "classical", or "llm"

        Returns:
            Dictionary with extraction results
        """
        if model_type == "auto":
            # Try LLM first, fallback to classical
            try:
                entities = self._extract_with_llm(text)
                extraction_method = "llm"
            except Exception:
                entities = self._extract_with_classical(text)
                extraction_method = "classical"
        elif model_type == "llm":
            entities = self._extract_with_llm(text)
            extraction_method = "llm"
        elif model_type == "classical":
            entities = self._extract_with_classical(text)
            extraction_method = "classical"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Determine document type
        doc_type = self._classify_document_type(text)

        return {
            "text": text,
            "extraction_method": extraction_method,
            "entities": entities,
            "document_type": doc_type,
            "confidence_scores": self._calculate_confidence_scores(entities)
        }

    def _extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using LLM models."""
        entities = []

        # Try OpenAI first
        if self.openai_client and "openai_ner" in self.models:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.models["openai_ner"]["model_id"],
                    messages=[
                        {"role": "system", "content": "Extract named entities from the document. Return in format: [LABEL]entity_text[/LABEL]"},
                        {"role": "user", "content": text}
                    ]
                )
                response_text = response.choices[0].message.content
                entities = self._parse_llm_response(response_text, text)
            except Exception as e:
                print(f"OpenAI extraction failed: {e}")

        # Fallback to HuggingFace
        if not entities and "hf_ner" in self.models:
            try:
                ner_pipeline = self.models["hf_ner"]
                predictions = ner_pipeline(text)
                entities = self._convert_hf_predictions(predictions)
            except Exception as e:
                print(f"HuggingFace extraction failed: {e}")

        return entities

    def _extract_with_classical(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using classical ML models."""
        entities = []

        # Try CRF first
        if "crf_ner" in self.models:
            entities = self._extract_with_crf(text)

        # Fallback to Random Forest if no CRF
        if not entities:
            entities = self._extract_with_rf(text)

        return entities

    def _extract_with_crf(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using CRF model."""
        entities = []

        if "crf_ner" not in self.models:
            return entities

        try:
            # Tokenize text (simple whitespace split for demo)
            tokens = text.split()

            # Get predictions (mock implementation)
            crf_model = self.models["crf_ner"]
            # In real implementation, you'd prepare features for each token
            # predictions = crf_model.predict([features])[0]

            # Mock predictions for demo
            predictions = ["O"] * len(tokens)
            if "invoice" in text.lower():
                predictions[0] = "B-DOC_TYPE"
            if "$" in text:
                dollar_idx = text.find("$")
                token_idx = len(text[:dollar_idx].split()) - 1
                if token_idx < len(predictions):
                    predictions[token_idx] = "B-AMOUNT"

            # Convert to entities
            current_entity = None
            start_idx = 0
            for i, (token, pred) in enumerate(zip(tokens, predictions)):
                if pred.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    label = pred[2:]
                    current_entity = {
                        "text": token,
                        "label": label,
                        "start": text.find(token, start_idx),
                        "end": text.find(token, start_idx) + len(token),
                        "confidence": 0.8
                    }
                    start_idx = current_entity["end"]
                elif pred.startswith("I-") and current_entity:
                    current_entity["text"] += " " + token
                    current_entity["end"] = text.find(token, start_idx) + len(token)
                    start_idx = current_entity["end"]
                else:
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None

            if current_entity:
                entities.append(current_entity)

        except Exception as e:
            print(f"CRF extraction failed: {e}")

        return entities

    def _extract_with_rf(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using Random Forest models."""
        entities = []

        # Simple rule-based extraction as fallback
        if "invoice" in text.lower():
            start = text.lower().find("invoice")
            entities.append({
                "text": text[start:start+7],
                "label": "DOC_TYPE",
                "start": start,
                "end": start+7,
                "confidence": 0.7
            })

        # Extract amounts
        amounts = re.findall(r'\$[\d,]+\.?\d*', text)
        for amount in amounts:
            start = text.find(amount)
            entities.append({
                "text": amount,
                "label": "AMOUNT",
                "start": start,
                "end": start + len(amount),
                "confidence": 0.9
            })

        return entities

    def _classify_document_type(self, text: str) -> str:
        """Classify document type."""
        text_lower = text.lower()
        if "invoice" in text_lower:
            return "invoice"
        elif "receipt" in text_lower:
            return "receipt"
        elif "bill" in text_lower:
            return "bill"
        else:
            return "unknown"

    def _calculate_confidence_scores(self, entities: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence scores for extraction."""
        if not entities:
            return {"overall": 0.0}

        confidences = [e.get("confidence", 0.5) for e in entities]
        return {
            "overall": sum(confidences) / len(confidences),
            "entity_count": len(entities)
        }

    def _parse_llm_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into entities."""
        entities = []
        pattern = r'\[([^\]]+)\](.*?)\[/\1\]'
        matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)

        for label, entity_text in matches:
            # Find position in original text
            start = original_text.find(entity_text.strip())
            if start != -1:
                entities.append({
                    "text": entity_text.strip(),
                    "label": label.upper(),
                    "start": start,
                    "end": start + len(entity_text.strip()),
                    "confidence": 1.0
                })

        return entities

    def _convert_hf_predictions(self, predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert HuggingFace predictions to standard format."""
        entities = []
        for pred in predictions:
            entities.append({
                "text": pred["word"],
                "label": pred["entity_group"],
                "start": pred["start"],
                "end": pred["end"],
                "confidence": pred["score"]
            })
        return entities

    def retrain_on_feedback(self, predicted: Dict[str, Any], corrected: Dict[str, Any], text: str) -> bool:
        """
        Retrain models based on user corrections.

        Args:
            predicted: Model predictions
            corrected: User corrections
            text: Original document text

        Returns:
            True if retraining was successful
        """
        # This is a placeholder for continuous learning
        # In a real implementation, you would:
        # 1. Add the corrected example to training data
        # 2. Retrain the models periodically
        # 3. Update model versions

        print("Retraining on feedback - placeholder implementation")
        print(f"Text: {text[:100]}...")
        print(f"Predicted: {predicted}")
        print(f"Corrected: {corrected}")

        return True


def main():
    """Command-line interface for testing the extractor."""
    import argparse

    parser = argparse.ArgumentParser(description="Test custom document extractor")
    parser.add_argument("--text", help="Document text to extract from")
    parser.add_argument("--file", help="File containing document text")
    parser.add_argument("--model-type", choices=["auto", "classical", "llm"], default="auto",
                       help="Type of model to use")

    args = parser.parse_args()

    # Get text
    text = ""
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("Please provide --text or --file")
        return

    # Extract information
    extractor = CustomExtractor()
    result = extractor.extract_document_info(text, args.model_type)

    # Print results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
