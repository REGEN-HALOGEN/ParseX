# TODO: Fix Errors in Specified Files

## Files to Fix:
- app/extraction/custom_extractor.py
- backend/scripts/label_data.py
- backend/scripts/retrain_on_corrections.py
- backend/scripts/train_classical_ml.py
- backend/scripts/train_llm.py
- src/api.py
- tests/test_custom_extractor.py
- backend/scripts/preprocess_data.py

## Identified Issues:
1. **custom_extractor.py**: Incomplete _load_llm_models method (NaN placeholder), missing methods like _extract_with_llm, _extract_with_classical, _extract_with_crf, extract_document_info incomplete.
2. **label_data.py**: NaN in _extract_potential_fields method.
3. **retrain_on_corrections.py**: NaN in load_corrections method.
4. **train_classical_ml.py**: NaN in train_crf method.
5. **train_llm.py**: NaN in prepare_ner_dataset and train_llm_pipeline methods.
6. **api.py**: Potential import issues if custom_extractor has errors.
7. **test_custom_extractor.py**: Incomplete test method (NaNest).
8. **preprocess_data.py**: NaN in extract_features method.

## Plan:
- Replace all "NaN" placeholders with appropriate code.
- Complete incomplete methods based on context and typical implementations.
- Ensure syntax is correct and imports work.
- Test compilation after fixes.

## Steps:
1. Fix custom_extractor.py: Complete _load_llm_models, add missing methods.
2. Fix label_data.py: Replace NaN in _extract_potential_fields.
3. Fix retrain_on_corrections.py: Replace NaN in load_corrections.
4. Fix train_classical_ml.py: Replace NaN in train_crf.
5. Fix train_llm.py: Replace NaN in prepare_ner_dataset and train_llm_pipeline.
6. Fix test_custom_extractor.py: Complete the test method.
7. Fix preprocess_data.py: Replace NaN in extract_features.
8. Check api.py for any issues.
9. Run py_compile on all files to verify fixes.
