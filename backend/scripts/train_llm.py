"""
Training script for LLM fine-tuning (OpenAI GPT or HuggingFace models) for custom extraction.
"""

import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from transformers import pipeline
import evaluate
import numpy as np


class LLMTrainer:
    """Trainer for LLM models."""

    def __init__(self, model_dir: str = "backend/models", cache_dir: str = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir

        # Initialize OpenAI client if API key is available
        self.openai_client = None
        if os.getenv('OPENAI_API_KEY'):
            self.openai_client = openai.OpenAI()

    def load_ner_data(self, train_file: str, val_file: str = None, test_file: str = None) -> DatasetDict:
        """
        Load NER data and convert to HuggingFace Dataset format.

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            test_file: Path to test data

        Returns:
            DatasetDict with train/val/test splits
        """
        def load_jsonl(file_path: str) -> List[Dict]:
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data

        # Load data
        train_data = load_jsonl(train_file)
        val_data = load_jsonl(val_file) if val_file else []
        test_data = load_jsonl(test_file) if test_file else []

        # Create datasets
        datasets = {"train": Dataset.from_list(train_data)}
        if val_data:
            datasets["validation"] = Dataset.from_list(val_data)
        if test_data:
            datasets["test"] = Dataset.from_list(test_data)

        return DatasetDict(datasets)

    def prepare_ner_dataset(self, dataset_dict: DatasetDict, model_name: str = "distilbert-base-uncased") -> DatasetDict:
        """
        Prepare NER dataset for training with tokenization and label alignment.

        Args:
            dataset_dict: Raw dataset
            model_name: HuggingFace model name

        Returns:
            Processed DatasetDict ready for training
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)

        # Get label list
        label_list = self._get_label_list(dataset_dict)
        label_to_id = {label: i for i, label in enumerate(label_list)}

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                is_split_into_words=False
            )

            labels = []
            for i, label in enumerate(examples["entities"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # Start of a new word
                        entity_labels = [entity["label"] for entity in label if entity["start"] <= word_idx < entity["end"]]
                        label_ids.append(label_to_id.get(entity_labels[0], label_to_id["O"]) if entity_labels else label_to_id["O"])
                    else:
                        # Continuation of the same word
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)

            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        # Process datasets
        processed_datasets = {}
        for split, dataset in dataset_dict.items():
            processed_datasets[split] = dataset.map(tokenize_and_align_labels, batched=True)

        return DatasetDict(processed_datasets)

    def _get_label_list(self, dataset_dict: DatasetDict) -> List[str]:
        """Extract unique labels from the dataset."""
        labels = set()
        for dataset in dataset_dict.values():
            for example in dataset:
                for entity in example.get("entities", []):
                    labels.add(entity["label"])
                    labels.add(f"B-{entity['label']}")
                    labels.add(f"I-{entity['label']}")
        labels.add("O")
        return sorted(list(labels))

    def train_huggingface_model(self, dataset_dict: DatasetDict, model_name: str = "distilbert-base-uncased",
                              num_epochs: int = 3, batch_size: int = 8) -> str:
        """
        Train a HuggingFace NER model.

        Args:
            dataset_dict: Processed dataset
            model_name: Model name
            num_epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Path to saved model
        """
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(self._get_label_list(dataset_dict)),
            cache_dir=self.cache_dir
        )

        training_args = TrainingArguments(
            output_dir=str(self.model_dir / "hf_ner"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )

        data_collator = DataCollatorForTokenClassification(tokenizer=AutoTokenizer.from_pretrained(model_name))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict.get("validation"),
            data_collator=data_collator,
            compute_metrics=self._compute_metrics,
        )

        trainer.train()

        # Save model
        model_path = self.model_dir / "hf_ner.pkl"
        pipeline_obj = pipeline(
            "ner",
            model=trainer.model,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            aggregation_strategy="simple"
        )
        joblib.dump(pipeline_obj, model_path)

        return str(model_path)

    def _compute_metrics(self, p):
        """Compute metrics for evaluation."""
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self._get_label_list(None)[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._get_label_list(None)[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        metric = evaluate.load("seqeval")
        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def fine_tune_openai(self, train_file: str, model_name: str = "gpt-3.5-turbo", num_epochs: int = 3) -> str:
        """
        Fine-tune an OpenAI model.

        Args:
            train_file: Path to training data
            model_name: OpenAI model name
            num_epochs: Number of epochs

        Returns:
            Fine-tuned model ID
        """
        if not self.openai_client:
            raise ValueError("OpenAI API key not found")

        # Convert training data to OpenAI format
        training_data = self._convert_to_openai_format(train_file)

        # Upload training file
        with open("temp_training.jsonl", "w") as f:
            for example in training_data:
                f.write(json.dumps(example) + "\n")

        with open("temp_training.jsonl", "rb") as f:
            file_response = self.openai_client.files.create(file=f, purpose="fine-tune")

        # Start fine-tuning
        fine_tune_response = self.openai_client.fine_tuning.jobs.create(
            training_file=file_response.id,
            model=model_name,
            hyperparameters={"n_epochs": num_epochs}
        )

        # Save model info
        model_info = {
            "job_id": fine_tune_response.id,
            "model_id": fine_tune_response.fine_tuned_model,
            "status": "pending"
        }

        model_path = self.model_dir / "openai_ner.json"
        with open(model_path, "w") as f:
            json.dump(model_info, f)

        # Clean up
        os.remove("temp_training.jsonl")

        return fine_tune_response.id

    def _convert_to_openai_format(self, train_file: str) -> List[Dict[str, Any]]:
        """Convert training data to OpenAI fine-tuning format."""
        training_data = []
        with open(train_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                text = data["text"]
                entities = data.get("entities", [])

                # Create labeled text
                labeled_text = text
                offset = 0
                for entity in sorted(entities, key=lambda x: x["start"]):
                    start, end, label = entity["start"], entity["end"], entity["label"]
                    labeled_text = (
                        labeled_text[:start + offset] +
                        f"[{label}]" + labeled_text[start + offset:end + offset] + f"[/{label}]" +
                        labeled_text[end + offset:]
                    )
                    offset += len(f"[{label}][/{label}]")

                training_data.append({
                    "messages": [
                        {"role": "system", "content": "Extract named entities from the document."},
                        {"role": "user", "content": text},
                        {"role": "assistant", "content": labeled_text}
                    ]
                })

        return training_data

    def train_llm_pipeline(self, train_file: str, val_file: str = None, test_file: str = None,
                         model_type: str = "huggingface", model_name: str = "distilbert-base-uncased",
                         num_epochs: int = 3, batch_size: int = 8) -> str:
        """
        Complete LLM training pipeline.

        Args:
            train_file: Path to training data
            val_file: Path to validation data
            test_file: Path to test data
            model_type: "huggingface" or "openai"
            model_name: Model name
            num_epochs: Number of epochs
            batch_size: Batch size

        Returns:
            Path to trained model or job ID
        """
        if model_type == "openai":
            return self.fine_tune_openai(train_file, model_name, num_epochs)
        else:
            # Load and prepare data
            raw_dataset = self.load_ner_data(train_file, val_file, test_file)
            processed_dataset = self.prepare_ner_dataset(raw_dataset, model_name)

            # Train model
            model_path = self.train_huggingface_model(
                processed_dataset, model_name, num_epochs, batch_size
            )

            return model_path


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train LLM models for custom extraction")
    parser.add_argument("--train-file", required=True, help="Path to training data (JSONL)")
    parser.add_argument("--val-file", help="Path to validation data (JSONL)")
    parser.add_argument("--test-file", help="Path to test data (JSONL)")
    parser.add_argument("--model-type", choices=["huggingface", "openai"], default="huggingface",
                       help="Type of model to train")
    parser.add_argument("--model-name", default="distilbert-base-uncased",
                       help="Model name (HuggingFace or OpenAI)")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--model-dir", default="backend/models", help="Directory to save models")

    args = parser.parse_args()

    # Initialize trainer
    trainer = LLMTrainer(args.model_dir)

    # Train model
    result = trainer.train_llm_pipeline(
        args.train_file,
        args.val_file,
        args.test_file,
        args.model_type,
        args.model_name,
        args.num_epochs,
        args.batch_size
    )

    print(f"\nTraining complete! Result: {result}")


if __name__ == "__main__":
    main()
