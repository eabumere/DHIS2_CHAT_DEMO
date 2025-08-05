# utils/ocr_document_parsing.py

import re
import pdfplumber
import pytesseract
import fitz  # PyMuPDF
from PIL import Image
import PyPDF2
import os

def is_scanned_pdf(filepath: str) -> bool:
    """
    Determines whether a PDF is scanned (i.e., has no extractable text).
    """
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                return False
    return True

def ocr_from_pdf(pdf_path: str) -> str:
    """
    Runs OCR on each page of a scanned PDF.
    """
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(img)
        texts.append(text)
    return "\n".join(texts)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a text-based (non-scanned) PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def ocr_from_image(image_path: str) -> str:
    """
    Extracts text from an image using OCR.
    """
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)

def extract_data_from_text(text: str) -> list:
    """
    Parses structured data from raw OCR or extracted text.
    """
    entries = []
    for match in re.finditer(
        r"dataElement:\s*(\w+).*?period:\s*(\d+).*?orgUnit:\s*(\w+).*?value:\s*(\d+)",
        text,
        re.DOTALL,
    ):
        entries.append({
            "dataElement": match.group(1),
            "period": match.group(2),
            "orgUnit": match.group(3),
            "categoryOptionCombo": "default",
            "attributeOptionCombo": "default",
            "value": match.group(4)
        })
    return entries
