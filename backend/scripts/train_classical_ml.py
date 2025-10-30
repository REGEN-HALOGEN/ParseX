"""
Training script for classical ML models (CRF, Random Forest) for custom extraction.
"""

import json
import argparse
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
import joblib


class ClassicalMLTrainer:
    """Trainer for classical ML models."""

    def __init__(self, model_dir: str = "backend/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)

    def load_training_data(self, train_file: str, val_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and validation data."""
        train_df = pd.read_csv(train_file)

        val_df = None
        if val_file:
            val_df = pd.read_csv(val_file)

        return train_df, val_df

    def train_random_forest(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None,
                          target_column: str = "document_type") -> RandomForestClassifier:
        """
        Train a Random Forest classifier.

        Args:
            train_df: Training data
            val_df: Validation data (optional)
            target_column: Column to predict

        Returns:
            Trained RandomForestClassifier
        """
        print(f"Training Random Forest for {target_column}...")

        # Prepare features and target
        feature_columns = [col for col in train_df.columns if col != target_column]
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        # Encode target if needed
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train_encoded)

        print(f"Best parameters: {grid_search.best_params_}")

        # Save encoder
        encoder_path = self.model_dir / f"rf_{target_column}_encoder.pkl"
        joblib.dump(le, encoder_path)

        # Evaluate on validation set
        if val_df is not None:
            X_val = val_df[feature_columns]
            y_val = le.transform(val_df[target_column])
            y_pred_val = grid_search.predict(X_val)

            print("Validation Results:")
            print(f"Accuracy: {accuracy_score(y_val, y_pred_val)}")
            print(classification_report(y_val, y_pred_val))

        # Save model
        model_path = self.model_dir / f"rf_{target_column}.pkl"
        joblib.dump(grid_search.best_estimator_, model_path)
        print(f"Random Forest model saved to {model_path}")

        return grid_search.best_estimator_

    def train_crf(self, train_file: str, val_file: str = None) -> sklearn_crfsuite.CRF:
        """
        Train a CRF model for NER.

        Args:
            train_file: Path to training data (JSONL format)
            val_file: Path to validation data (optional)

        Returns:
            Trained CRF model
        """
        print("Training CRF for NER...")

        # Load training data
        train_sentences = self._load_ner_data(train_file)

        # Prepare features
        X_train = [self._sent2features(s) for s in train_sentences]
        y_train = [self._sent2labels(s) for s in train_sentences]

        # Train CRF
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

        crf.fit(X_train, y_train)

        # Evaluate on validation set
        if val_file:
            val_sentences = self._load_ner_data(val_file)
            X_val = [self._sent2features(s) for s in val_sentences]
            y_val = [self._sent2labels(s) for s in val_sentences]
            y_pred_val = crf.predict(X_val)

            print("Validation Results:")
            print(metrics.flat_classification_report(y_val, y_pred_val))

        # Save model
        model_path = self.model_dir / "crf_ner.pkl"
        joblib.dump(crf, model_path)
        print(f"CRF model saved to {model_path}")

        return crf

    def _load_ner_data(self, file_path: str) -> List[List[Dict[str, Any]]]:
        """Load NER training data from JSONL file."""
        sentences = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                sentences.append(self._text_to_sentences(data))
        return sentences

    def _text_to_sentences(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert text and entities to sentence format."""
        text = data['text']
        entities = data.get('entities', [])

        # Simple tokenization (split by spaces)
        tokens = text.split()
        labels = ['O'] * len(tokens)

        # Assign labels based on entities
        for entity in entities:
            start, end, label = entity['start'], entity['end'], entity['label']
            entity_text = text[start:end]

            # Find token indices for this entity
            entity_tokens = entity_text.split()
            start_token_idx = None
            end_token_idx = None

            current_pos = 0
            for i, token in enumerate(tokens):
                if current_pos == start:
                    start_token_idx = i
                if current_pos + len(token) == end:
                    end_token_idx = i
                    break
                current_pos += len(token) + 1  # +1 for space

            if start_token_idx is not None and end_token_idx is not None:
                labels[start_token_idx] = f'B-{label}'
                for i in range(start_token_idx + 1, end_token_idx + 1):
                    labels[i] = f'I-{label}'

        return [{'token': token, 'label': label} for token, label in zip(tokens, labels)]

    def _sent2features(self, sent: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract features from a sentence for CRF."""
        features = []
        for i, token_dict in enumerate(sent):
            token = token_dict['token']
            features.append({
                'bias': 1.0,
                'token.lower()': token.lower(),
                'token.isupper()': token.isupper(),
                'token.istitle()': token.istitle(),
                'token.isdigit()': token.isdigit(),
                'token.length': len(token),
                'token.has_digit': any(c.isdigit() for c in token),
                'token.has_alpha': any(c.isalpha() for c in token),
                'BOS': i == 0,
                'EOS': i == len(sent) - 1,
            })
        return features

    def _sent2labels(self, sent: List[Dict[str, Any]]) -> List[str]:
        """Extract labels from a sentence."""
        return [token_dict['label'] for token_dict in sent]

    def train_all_models(self, classification_file: str, ner_file: str,
                        val_classification_file: str = None, val_ner_file: str = None) -> Dict[str, Any]:
        """
        Train all classical ML models.

        Args:
            classification_file: Path to classification training data
            ner_file: Path to NER training data
            val_classification_file: Path to validation classification data
            val_ner_file: Path to validation NER data

        Returns:
            Dictionary with trained models
        """
        models = {}

        # Load data
        train_df, val_df = self.load_training_data(classification_file, val_classification_file)

        # Train Random Forest for document type classification
        rf_model = self.train_random_forest(train_df, val_df, "document_type")
        models['document_type_rf'] = rf_model

        # Train CRF for NER
        crf_model = self.train_crf(ner_file, val_ner_file)
        models['ner_crf'] = crf_model

        return models


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train classical ML models for custom extraction")
    parser.add_argument("--classification-train", required=True, help="Path to classification training data (CSV)")
    parser.add_argument("--ner-train", required=True, help="Path to NER training data (JSONL)")
    parser.add_argument("--classification-val", help="Path to classification validation data (CSV)")
    parser.add_argument("--ner-val", help="Path to NER validation data (JSONL)")
    parser.add_argument("--model-dir", default="backend/models", help="Directory to save trained models")

    args = parser.parse_args()

    # Initialize trainer
    trainer = ClassicalMLTrainer(args.model_dir)

    # Train all models
    models = trainer.train_all_models(
        args.classification_train,
        args.ner_train,
        args.classification_val,
        args.ner_val
    )

    print(f"\nTraining complete! Models saved to {args.model_dir}")
    print("Trained models:", list(models.keys()))


if __name__ == "__main__":
    main()
