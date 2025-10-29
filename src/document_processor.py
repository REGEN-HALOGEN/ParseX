from typing import Dict, List, Optional
import importlib
import cv2
import pytesseract
from PIL import Image
import numpy as np


class DocumentProcessor:
    def __init__(self) -> None:
        self.image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None

        # detect backends
        self.has_tesseract: bool = self._check_tesseract()
        has_eas, module = self._check_easyocr()
        self.has_easyocr: bool = has_eas
        self.easyocr = module

    def _check_tesseract(self) -> bool:
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def _check_easyocr(self):
        try:
            module = importlib.import_module("easyocr")
            return True, module
        except Exception:
            return False, None

    def load_image(self, path: str) -> None:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        self.image = img
        self.processed_image = img.copy()

    def preprocess_image(self) -> None:
        if self.image is None:
            raise ValueError("No image loaded")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        self.processed_image = cv2.fastNlMeansDenoising(binary)

    def extract_text(self) -> str:
        if self.processed_image is None:
            raise ValueError("No processed image available")

        # Priority: Tesseract, then EasyOCR. If none, return empty string.
        if self.has_tesseract:
            pil = Image.fromarray(self.processed_image)
            return pytesseract.image_to_string(pil).strip()

        if not self.has_easyocr or self.easyocr is None:
            return ""

        # EasyOCR path
        rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB) if self.processed_image.ndim == 2 else cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        reader = self.easyocr.Reader(["en"], gpu=False)  # type: ignore
        results = reader.readtext(rgb)
        return "\n".join([r[1] for r in results]).strip()

    def detect_layout(self) -> Dict[str, List[Dict]]:
        if self.processed_image is None:
            raise ValueError("No processed image available")

        # Tesseract-based layout
        if self.has_tesseract:
            data = pytesseract.image_to_data(Image.fromarray(self.processed_image), output_type=pytesseract.Output.DICT)
            blocks: List[Dict] = []
            for i in range(len(data.get("text", []))):
                txt = (data.get("text", [])[i] or "").strip()
                if txt:
                    blocks.append({
                        "text": txt,
                        "confidence": data.get("conf", [])[i],
                        "position": {
                            "x": int(data.get("left", [])[i]),
                            "y": int(data.get("top", [])[i]),
                            "width": int(data.get("width", [])[i]),
                            "height": int(data.get("height", [])[i]),
                        },
                    })
            return {"blocks": blocks}

        # EasyOCR-based layout
        if not self.has_easyocr or self.easyocr is None:
            return {"blocks": []}

        rgb = cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2RGB) if self.processed_image.ndim == 2 else cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
        reader = self.easyocr.Reader(["en"], gpu=False)  # type: ignore
        results = reader.readtext(rgb)
        blocks: List[Dict] = []
        for bbox, text, conf in results:
            xs = [int(p[0]) for p in bbox]
            ys = [int(p[1]) for p in bbox]
            blocks.append({
                "text": text,
                "confidence": float(conf),
                "position": {
                    "x": min(xs),
                    "y": min(ys),
                    "width": max(xs) - min(xs),
                    "height": max(ys) - min(ys),
                },
            })
        return {"blocks": blocks}
