# Training Custom Extraction Models

This document provides a comprehensive guide for training custom extraction models for the AI Document Intelligence system.

## Overview

The system supports multiple approaches for training custom extraction models:

1. **Classical ML Models**: CRF (Conditional Random Fields) and Random Forest classifiers
2. **LLM Fine-tuning**: Fine-tuning pre-trained language models (HuggingFace or OpenAI)
3. **Continuous Learning**: Retraining models based on user corrections

## Prerequisites

### Software Requirements

- Python 3.8+
- Required packages (install via `pip install -r requirements.txt`)
- Tesseract OCR (for data collection)
- Poppler (for PDF processing) or PyMuPDF

### Hardware Requirements

- **Classical ML**: 4GB RAM minimum, CPU-based
- **LLM Fine-tuning**: 8GB+ RAM, GPU recommended for faster training
- **OpenAI Fine-tuning**: API key and credits required

## Data Collection

### Step 1: Collect Raw Documents

Use the data collection script to gather documents for training:

```bash
python backend/scripts/collect_training_data.py \
    --input-dir /path/to/document/images \
    --output-dir backend/training_data \
    --batch-size 10
```

**Parameters:**
- `--input-dir`: Directory containing document images (PNG, JPG, JPEG, TIFF, BMP)
- `--output-dir`: Where to save processed data
- `--batch-size`: Number of documents to process simultaneously
- `--extensions`: File extensions to process (default: common image formats)

**Output:** `training_data_YYYYMMDD_HHMMSS.json` containing extracted text and metadata.

### Step 2: Label Training Data

Use the labeling interface to create ground truth annotations:

```bash
python backend/scripts/label_data.py \
    --data-file backend/training_data/training_data_YYYYMMDD_HHMMSS.json \
    --output-file labeled_data.json
```

**Interactive Labeling Process:**
1. Review extracted text
2. Identify document type (invoice, receipt, bill, statement)
3. Label key fields:
   - Vendor name
   - Vendor address
   - GST/Tax ID
   - Invoice number
   - Dates (invoice date, due date)
   - Amounts (subtotal, tax, total)
   - Line items (description, quantity, price)

**Output:** `labeled_data.json` with ground truth annotations.

### Step 3: Preprocess Data

Convert labeled data into training format:

```bash
python backend/scripts/preprocess_data.py \
    --labeled-file backend/training_data/labeled_data.json \
    --output-dir backend/training_data
```

**Output Files:**
- `ner_train.jsonl`, `ner_val.jsonl`, `ner_test.jsonl`: NER training data
- `classification_train.csv`, `classification_val.csv`, `classification_test.csv`: Classification data

## Model Training

### Classical ML Training

Train CRF and Random Forest models:

```bash
python backend/scripts/train_classical_ml.py \
    --classification-train backend/training_data/classification_train.csv \
    --ner-train backend/training_data/ner_train.jsonl \
    --classification-val backend/training_data/classification_val.csv \
    --ner-val backend/training_data/ner_val.jsonl \
    --model-dir backend/models
```

**Output:**
- `rf_document_type.pkl`: Document type classifier
- `crf_ner.pkl`: NER model
- Encoder files for label mapping

### LLM Fine-tuning (HuggingFace)

Fine-tune a transformer model:

```bash
python backend/scripts/train_llm.py \
    --train-file backend/training_data/ner_train.jsonl \
    --val-file backend/training_data/ner_val.jsonl \
    --test-file backend/training_data/ner_test.jsonl \
    --model-type huggingface \
    --model-name distilbert-base-uncased \
    --num-epochs 3 \
    --batch-size 8 \
    --model-dir backend/models
```

**Recommended Models:**
- `distilbert-base-uncased` (fast, good performance)
- `microsoft/DialoGPT-medium` (for conversational documents)
- `bert-base-uncased` (higher accuracy, slower)

### LLM Fine-tuning (OpenAI)

Fine-tune a GPT model using OpenAI API:

```bash
export OPENAI_API_KEY="your-api-key-here"

python backend/scripts/train_llm.py \
    --train-file backend/training_data/ner_train.jsonl \
    --model-type openai \
    --model-name gpt-3.5-turbo \
    --num-epochs 3 \
    --model-dir backend/models
```

**Note:** Requires OpenAI API key and sufficient credits.

## Model Evaluation

### Evaluate Classical Models

The training scripts automatically evaluate on validation data and print metrics:

- **Classification**: Accuracy, precision, recall, F1-score
- **NER**: F1-score, precision, recall per entity type

### Evaluate LLM Models

HuggingFace models show training/validation metrics during training:
- Loss
- F1-score, precision, recall
- Accuracy

OpenAI models can be evaluated using their playground or by running inference on test data.

## Continuous Learning

### Adding User Corrections

Use the continuous learning script to incorporate user feedback:

```bash
python backend/scripts/retrain_on_corrections.py \
    --action add \
    --text "Original document text..." \
    --original original_extraction.json \
    --corrected corrected_extraction.json \
    --user-id username
```

### Processing Corrections

Batch process accumulated corrections:

```bash
python backend/scripts/retrain_on_corrections.py \
    --action process \
    --min-corrections 10
```

This will:
1. Convert corrections to training format
2. Merge with existing training data
3. Retrain models
4. Backup old models

### Monitoring Feedback

Check feedback statistics:

```bash
python backend/scripts/retrain_on_corrections.py --action stats
```

## Deployment

### Model Loading

The `CustomExtractor` class automatically loads all available models from `backend/models/`:

```python
from app.extraction.custom_extractor import CustomExtractor

extractor = CustomExtractor()
result = extractor.extract_document_info(document_text)
```

### API Integration

Models are accessible via the FastAPI endpoint:

```bash
curl -X POST "http://localhost:8000/extract" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@document.png" \
     -F "model_type=auto"
```

### Streamlit Interface

The web interface provides:
- Document upload
- Custom extraction with trained models
- Correction interface for continuous learning

## Best Practices

### Data Quality

1. **Diverse Documents**: Include various document types, formats, and qualities
2. **Consistent Labeling**: Use clear labeling guidelines
3. **Quality Control**: Review and validate labels before training

### Training Tips

1. **Start Small**: Begin with 50-100 labeled documents
2. **Iterate**: Train, evaluate, collect more data, retrain
3. **Balance Classes**: Ensure balanced representation of document types
4. **Cross-validation**: Use cross-validation for hyperparameter tuning

### Model Selection

1. **Classical ML**: Good for structured data, fast inference, works offline
2. **LLM Fine-tuning**: Better for complex patterns, handles variations well
3. **Hybrid Approach**: Use both - classical for speed, LLM for accuracy

### Performance Optimization

1. **Batch Processing**: Process documents in batches during training
2. **GPU Usage**: Use GPU for LLM training when available
3. **Model Quantization**: Reduce model size for deployment
4. **Caching**: Cache preprocessed data to speed up iterations

## Troubleshooting

### Common Issues

1. **OCR Quality**: Poor OCR leads to bad training data
   - Solution: Improve image preprocessing, use better OCR engines

2. **Label Inconsistency**: Inconsistent labeling across documents
   - Solution: Create detailed labeling guidelines, review labels

3. **Overfitting**: Model performs well on training data but poorly on new documents
   - Solution: More diverse training data, regularization, cross-validation

4. **Memory Issues**: Large models don't fit in memory
   - Solution: Use smaller models, gradient checkpointing, or CPU training

### Getting Help

1. Check the logs for detailed error messages
2. Validate input data formats
3. Test with smaller datasets first
4. Monitor system resources during training

## Advanced Topics

### Custom Entity Types

To add new entity types:

1. Update labeling schema in `label_data.py`
2. Add new labels to training data
3. Retrain models
4. Update extraction logic if needed

### Multi-language Support

For multiple languages:

1. Use multilingual OCR engines (EasyOCR)
2. Choose multilingual base models (mBERT, XLM-R)
3. Include diverse language documents in training

### Domain Adaptation

For specific domains:

1. Fine-tune on domain-specific documents
2. Use domain-specific preprocessing
3. Include domain terminology in training data

This training pipeline transforms the basic document intelligence system into a powerful, customizable extraction platform that improves over time through continuous learning.
