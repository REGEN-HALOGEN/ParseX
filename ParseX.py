"""
Streamlit web interface for the AI Document Intelligence system.
"""

import streamlit as st
import os
import tempfile
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from src.document_processor import DocumentProcessor
from src.text_analyzer import TextAnalyzer

# Optional PyMuPDF fallback for PDF -> image when poppler/pdf2image is unavailable
try:
    import fitz  # PyMuPDF
    _HAS_PYMUPDF = True
except Exception:
    fitz = None
    _HAS_PYMUPDF = False

# Optional libraries (import when available)
try:
    from pdf2image import convert_from_bytes
    _HAS_PDF2IMAGE = True
except Exception:
    convert_from_bytes = None
    _HAS_PDF2IMAGE = False

try:
    from docx import Document as DocxDocument
    _HAS_PYTHON_DOCX = True
except Exception:
    DocxDocument = None
    _HAS_PYTHON_DOCX = False


@st.cache_resource
def get_processors():
    """Create and cache processor instances for the Streamlit session."""
    return DocumentProcessor(), TextAnalyzer()


def _convert_pdf_bytes_to_images(raw_bytes):
    """Convert PDF bytes to a list of PIL.Image pages.

    Tries pdf2image.convert_from_bytes first (requires poppler). If that
    fails or isn't available, falls back to PyMuPDF (fitz) if installed.
    Raises RuntimeError with an actionable message when conversion isn't possible.
    """
    if raw_bytes is None:
        raise RuntimeError('No PDF bytes provided')

    # Try pdf2image first
    if _HAS_PDF2IMAGE and convert_from_bytes is not None:
        try:
            pages = convert_from_bytes(raw_bytes)
            if pages:
                return pages
        except Exception as e:
            # keep exception for later if no fallback
            pdf2_err = e
    else:
        pdf2_err = None

    # Fallback to PyMuPDF if available
    if _HAS_PYMUPDF and fitz is not None:
        try:
            pages = []
            doc = fitz.open(stream=raw_bytes, filetype='pdf')
            for p in doc:
                pix = p.get_pixmap(matrix=fitz.Matrix(2, 2))
                mode = 'RGBA' if pix.alpha else 'RGB'
                img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
                if mode == 'RGBA':
                    img = img.convert('RGB')
                pages.append(img)
            if pages:
                return pages
        except Exception as e:
            pymupdf_err = e
        # if pymupdf failed, fall through to raising combined error
    else:
        pymupdf_err = None

    # If we reach here, build a helpful error message
    parts = []
    if pdf2_err:
        parts.append(f'pdf2image error: {pdf2_err}')
    if pymupdf_err:
        parts.append(f'PyMuPDF error: {pymupdf_err}')
    parts.append('Install poppler (for pdf2image) or install PyMuPDF (pip install pymupdf)')
    raise RuntimeError('Unable to convert PDF to images. ' + ' | '.join(parts))


def main():
    """Main function for the Streamlit app."""
    doc_processor, text_analyzer = get_processors()
    st.title("AI Document Intelligence")
    st.write("Upload a document image to analyze its content.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg", "pdf", "docx"])

    # Initialize variables used later so they exist even if no file is uploaded
    raw_bytes = None
    extracted_text = None
    is_image = False
    is_docx = False

    if uploaded_file is not None:
        # Validate and display the uploaded file (image / PDF / DOCX)
        raw_bytes = None
        extracted_text = None
        is_image = False
        is_docx = False
        try:
            raw_bytes = uploaded_file.getvalue()
            # Try to open as an image
            pil_image = Image.open(BytesIO(raw_bytes))
            pil_image.load()
            st.image(pil_image, caption='Uploaded Image', use_container_width=True)
            is_image = True
        except UnidentifiedImageError:
            # Not a plain image — check PDF or DOCX
            filename = getattr(uploaded_file, 'name', '') or ''
            lower = filename.lower()
            if lower.endswith('.pdf'):
                # Always try to convert the PDF to images using our helper which
                # will use pdf2image (poppler) or PyMuPDF as a fallback. This
                # ensures we behave the same regardless of which backend is
                # available.
                try:
                    if raw_bytes is None:
                        st.error('Unable to read uploaded PDF bytes')
                    else:
                        st.info('Converting PDF to images (this may take a moment)...')
                        print('DEBUG: starting PDF -> images conversion')
                        pages = _convert_pdf_bytes_to_images(raw_bytes)
                        print(f'DEBUG: PDF conversion produced {len(pages)} pages')

                    page_count = len(pages)
                    st.write(f"PDF detected — {page_count} page(s) found.")

                    # Show a selector for pages
                    page_idx = st.select_slider("Choose page to display / process", options=list(range(1, page_count + 1)), value=1)
                    selected_index = page_idx - 1
                    pil_image = pages[selected_index]
                    # Store pages into session_state so we don't re-convert on reruns
                    st.session_state.setdefault('pdf_pages', pages)
                    st.session_state['pdf_selected_index'] = selected_index

                    # Display the selected page
                    st.image(pil_image, caption=f'PDF page {page_idx}', use_container_width=True)
                    is_image = True
                    # mark as pdf-derived image
                    st.session_state['is_pdf_upload'] = True
                except Exception as e:
                    print('ERROR: PDF conversion failed:', e)
                    st.error('Unable to convert PDF to image. Ensure poppler is installed and available on PATH, or install PyMuPDF. Error: ' + str(e))
            elif lower.endswith('.docx'):
                if not _HAS_PYTHON_DOCX:
                    st.error('DOCX support requires python-docx. Install python-docx to enable DOCX uploads.')
                else:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as _tmp:
                            if raw_bytes is None:
                                st.error('Unable to read uploaded DOCX bytes')
                                tmp_docx = _tmp.name
                            else:
                                _tmp.write(raw_bytes)
                                tmp_docx = _tmp.name
                        if _HAS_PYTHON_DOCX and raw_bytes is not None and DocxDocument is not None:
                            doc = DocxDocument(tmp_docx)
                            paragraphs = [p.text for p in doc.paragraphs if p.text]
                            extracted_text = "\n".join(paragraphs)
                        is_docx = True
                        if extracted_text and extracted_text.strip():
                            st.subheader('Extracted Text (from DOCX)')
                            st.write(extracted_text)
                        else:
                            st.warning('No text found inside the uploaded DOCX file')
                    except Exception as e:
                        st.error('Failed to read DOCX file: ' + str(e))
                    finally:
                        try:
                            os.unlink(tmp_docx)
                        except Exception:
                            pass
            else:
                st.error('Uploaded file is not a supported image/PDF/DOCX. Please upload PNG/JPEG, PDF, or DOCX files.')
        except Exception as e:
            st.error(f'Error reading uploaded file: {e}')

    # Process button
    if (is_image or is_docx) and st.button('Process Document'):
        with st.spinner('Processing document...'):
            try:
                # If DOCX, analyze extracted text directly
                if is_docx:
                    text = extracted_text or ''
                    layout = {'blocks': []}
                else:
                    temp_path = None
                    # If the image came from a PDF, get the selected page from session_state
                    if st.session_state.get('is_pdf_upload') and st.session_state.get('pdf_pages'):
                        pages = st.session_state['pdf_pages']
                        selected_index = st.session_state.get('pdf_selected_index', 0)
                        pil_to_process = pages[selected_index]

                        # Save that PIL image to a temporary PNG file for OpenCV
                        tmp_img_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                        try:
                            pil_to_process.save(tmp_img_file.name, format='PNG')
                            temp_path = tmp_img_file.name
                        finally:
                            tmp_img_file.close()
                    else:
                        # Save uploaded file bytes to a temp file and process
                        suffix = os.path.splitext(getattr(uploaded_file, 'name', ''))[1] if hasattr(uploaded_file, 'name') else ''
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        try:
                            tmp_file.write(raw_bytes or b'')
                            temp_path = tmp_file.name
                        finally:
                            tmp_file.close()

                    # Process document image
                    try:
                        if temp_path is None:
                            raise RuntimeError('No image data available to process')

                        doc_processor.load_image(temp_path)
                        doc_processor.preprocess_image()

                        # Extract text
                        text = doc_processor.extract_text()
                    except RuntimeError as e:
                        # Surface user-friendly message for missing Tesseract
                        if 'Tesseract' in str(e):
                            st.error('Tesseract OCR is not installed. Please install Tesseract to enable text extraction.')
                            text = ''
                        else:
                            st.error(f'Error processing document: {str(e)}')
                            text = ''
                    finally:
                        # Clean up temporary file for image-based processing
                        if temp_path:
                            try:
                                os.unlink(temp_path)
                            except Exception:
                                pass

                # If we have text, analyze and display results
                if text and text.strip():
                    st.subheader('Extracted Text')
                    st.write(text)

                    # Analyze text
                    analysis = text_analyzer.analyze_text(text)

                    # Display entities
                    st.subheader('Named Entities')
                    entities_df = []
                    for entity in analysis.get('entities', []):
                        entities_df.append({'Text': entity.get('text'), 'Type': entity.get('label')})
                    if entities_df:
                        st.dataframe(entities_df)
                    else:
                        st.write('No entities found')

                    # Display key phrases
                    st.subheader('Key Phrases')
                    key_phrases = analysis.get('key_phrases', [])
                    if key_phrases:
                        st.write(', '.join(key_phrases))
                    else:
                        st.write('No key phrases found')

                    # Display summary
                    st.subheader('Summary')
                    st.write(analysis.get('summary', ''))

                    # Display sentiment
                    st.subheader('Sentiment Analysis')
                    sentiment = analysis.get('sentiment', {'positive': 0.5, 'negative': 0.5})
                    positive = sentiment.get('positive', 0.5) * 100
                    negative = sentiment.get('negative', 0.5) * 100
                    st.write(f'Positive: {positive:.1f}%')
                    st.progress(sentiment.get('positive', 0.5))
                    st.write(f'Negative: {negative:.1f}%')
                    st.progress(sentiment.get('negative', 0.5))

                    # Display layout information (only for images)
                    if not is_docx:
                        st.subheader('Document Layout')
                        try:
                            layout = doc_processor.detect_layout()
                        except Exception as e:
                            # Avoid crashing the UI when layout detection isn't
                            # available (missing Tesseract/EasyOCR). Log the
                            # error and continue with an empty layout.
                            print('WARNING: layout detection failed:', e)
                            st.warning('Layout detection not available: ' + str(e))
                            layout = {'blocks': []}

                        for block in layout.get('blocks', []):
                            with st.expander(f"Block: {block.get('text', '')[:50]}..."):
                                st.write(f"Confidence: {block.get('confidence')}")
                                st.write('Position:')
                                st.json(block.get('position', {}))
                else:
                    st.warning('No text was extracted from the document')
            except Exception as e:
                st.error(f'Error processing file: {str(e)}')

    # Display helpful information in the sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
        This application uses AI to analyze document images and extract meaningful information:

        - Text extraction using OCR
        - Named entity recognition
        - Key phrase extraction
        - Text summarization
        - Sentiment analysis
        - Document layout analysis
        """)

        st.header("Supported Features")
        st.write("""
        - Image formats: PNG, JPG, JPEG, PDF
        - Multiple languages (through OCR)
        - Layout detection
        - Entity recognition
        """)


if __name__ == "__main__":
    main()