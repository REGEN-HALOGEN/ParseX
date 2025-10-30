"""
Script for retraining models on user corrections and feedback.
Implements continuous learning for the custom extraction system.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
import shutil


class ContinuousLearner:
    """Handles continuous learning by retraining on user corrections."""

    def __init__(self, training_data_dir: str = "backend/training_data",
                 model_dir: str = "backend/models"):
        self.training_data_dir = Path(training_data_dir)
        self.model_dir = Path(model_dir)
        self.corrections_file = self.training_data_dir / "user_corrections.jsonl"

        # Ensure directories exist
        self.training_data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)

    def add_correction(self, original_text: str, original_extraction: Dict[str, Any],
                      corrected_extraction: Dict[str, Any], user_id: str = "anonymous") -> str:
        """
        Add a user correction to the feedback dataset.

        Args:
            original_text: Original document text
            original_extraction: What the model extracted originally
            corrected_extraction: User's corrections
            user_id: Identifier for the user providing feedback

        Returns:
            Correction ID
        """
        correction_id = f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}"

        correction_entry = {
            "correction_id": correction_id,
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "original_text": original_text,
            "original_extraction": original_extraction,
            "corrected_extraction": corrected_extraction,
            "processed": False
        }

        # Append to corrections file
        with open(self.corrections_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(correction_entry, ensure_ascii=False) + '\n')

        print(f"Added correction: {correction_id}")
        return correction_id

    def load_corrections(self, processed: bool = False) -> List[Dict[str, Any]]:
        """
        Load corrections from the feedback file.

        Args:
            processed: Whether to load processed or unprocessed corrections

        Returns:
            List of correction entries
        """
        corrections = []
        if not self.corrections_file.exists():
            return corrections

        with open(self.corrections_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    correction = json.loads(line.strip())
                    if processed is None or correction.get("processed", False) == processed:
                        corrections.append(correction)
                except json.JSONDecodeError:
                    continue

        return corrections

    def process_corrections(self, min_corrections: int = 10) -> bool:
        """
        Process accumulated corrections and retrain models if enough data.

        Args:
            min_corrections: Minimum number of corrections needed for retraining

        Returns:
            True if retraining was performed
        """
        unprocessed_corrections = self.load_corrections(processed=False)

        if len(unprocessed_corrections) < min_corrections:
            print(f"Not enough corrections for retraining. Need {min_corrections}, have {len(unprocessed_corrections)}")
            return False

        print(f"Processing {len(unprocessed_corrections)} corrections...")

        # Add corrections to training data
        self._add_corrections_to_training_data(unprocessed_corrections)

        # Retrain models (placeholder - would call training scripts)
        success = self._retrain_models()

        if success:
            # Mark corrections as processed
            self._mark_corrections_processed(unprocessed_corrections)

        return success

    def _add_corrections_to_training_data(self, corrections: List[Dict[str, Any]]) -> None:
        """Add corrections to the training dataset."""
        training_file = self.training_data_dir / "training_data.jsonl"

        with open(training_file, 'a', encoding='utf-8') as f:
            for correction in corrections:
                # Convert correction to training format
                example = {
                    "text": correction["original_text"],
                    "entities": []
                }

                # Add corrected entities
                corrected = correction.get("corrected_extraction", {})
                entities = corrected.get("entities", [])
                for entity in entities:
                    if isinstance(entity, dict):
                        example["entities"].append({
                            "start": entity.get("start", 0),
                            "end": entity.get("end", 0),
                            "label": entity.get("label", "UNKNOWN"),
                            "text": entity.get("text", "")
                        })

                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"Added {len(corrections)} corrections to training data")

    def _retrain_models(self) -> bool:
        """Retrain the models with new data."""
        # This is a placeholder - in practice, you would:
        # 1. Call the training scripts
        # 2. Update model versions
        # 3. Backup old models

        print("Retraining models (placeholder implementation)")
        return True

    def _mark_corrections_processed(self, corrections: List[Dict[str, Any]]) -> None:
        """Mark corrections as processed."""
        # Read all corrections
        all_corrections = self.load_corrections(processed=None)

        # Update processed status
        for correction in all_corrections:
            if correction["correction_id"] in [c["correction_id"] for c in corrections]:
                correction["processed"] = True

        # Rewrite the file
        with open(self.corrections_file, 'w', encoding='utf-8') as f:
            for correction in all_corrections:
                f.write(json.dumps(correction, ensure_ascii=False) + '\n')

        print(f"Marked {len(corrections)} corrections as processed")

    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about user feedback and corrections.

        Returns:
            Dictionary with feedback statistics
        """
        all_corrections = self.load_corrections(processed=None)

        stats = {
            "total_corrections": len(all_corrections),
            "processed_corrections": len([c for c in all_corrections if c.get("processed", False)]),
            "unprocessed_corrections": len([c for c in all_corrections if not c.get("processed", False)]),
            "unique_users": len(set(c.get("user_id", "anonymous") for c in all_corrections)),
            "corrections_by_type": {}
        }

        # Analyze correction types
        for correction in all_corrections:
            corr_type = correction.get("corrected_extraction", {}).get("document_type", "unknown")
            stats["corrections_by_type"][corr_type] = stats["corrections_by_type"].get(corr_type, 0) + 1

        return stats


def main():
    """Command-line interface for continuous learning."""
    parser = argparse.ArgumentParser(description="Continuous learning for custom extraction models")
    parser.add_argument("--action", choices=["add", "process", "stats"], required=True,
                       help="Action to perform")
    parser.add_argument("--text", help="Original document text (for add action)")
    parser.add_argument("--original", help="JSON file with original extraction (for add action)")
    parser.add_argument("--corrected", help="JSON file with corrected extraction (for add action)")
    parser.add_argument("--user-id", default="cli", help="User ID for feedback")
    parser.add_argument("--min-corrections", type=int, default=10,
                       help="Minimum corrections needed for retraining")

    args = parser.parse_args()

    learner = ContinuousLearner()

    if args.action == "add":
        if not all([args.text, args.original, args.corrected]):
            print("For 'add' action, provide --text, --original, and --corrected")
            return

        # Load JSON files
        with open(args.original, 'r') as f:
            original = json.load(f)
        with open(args.corrected, 'r') as f:
            corrected = json.load(f)

        correction_id = learner.add_correction(args.text, original, corrected, args.user_id)
        print(f"Added correction: {correction_id}")

    elif args.action == "process":
        success = learner.process_corrections(args.min_corrections)
        if success:
            print("Retraining completed successfully")
        else:
            print("Retraining not performed (not enough corrections or error)")

    elif args.action == "stats":
        stats = learner.get_feedback_stats()
        print("Feedback Statistics:")
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
