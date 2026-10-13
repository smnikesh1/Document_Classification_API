
import os
import pytesseract

from PIL import Image, ImageOps, ImageEnhance
from pdf2image import convert_from_path




pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# Extract text from an image


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.

    Supported formats include PNG, JPG, JPEG, and TIFF.
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    try:
        with Image.open(image_path) as original_image:
            image = ImageOps.exif_transpose(original_image)
            image = image.convert("RGB")

            # Convert to grayscale for OCR
            image = ImageOps.grayscale(image)

            # Improve contrast
            image = ImageEnhance.Contrast(image).enhance(1.5)

            # OCR configuration: assume a block of text
            config = "--oem 3 --psm 6"

            text = pytesseract.image_to_string(
                image,
                lang="eng",
                config=config
            )

            return text.strip()

    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(
            "Tesseract OCR was not found. "
            "Check the installation path in app/ocr.py."
        ) from error


# 
# Extract text from a PDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Convert PDF pages into images and extract their text.

    Note: pdf2image also requires Poppler on Windows.
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    try:
        pages = convert_from_path(pdf_path)
        extracted_pages = []

        for page_number, page in enumerate(pages, start=1):
            page = page.convert("RGB")

            text = pytesseract.image_to_string(
                page,
                lang="eng",
                config="--oem 3 --psm 6"
            )

            extracted_pages.append(
                f"--- Page {page_number} ---\n{text.strip()}"
            )

        return "\n\n".join(extracted_pages).strip()

    except Exception as error:
        raise RuntimeError(
            f"Could not extract text from PDF: {error}"
        ) from error


# 
# Main function used by the project

def extract_text(file_path: str) -> str:
    """
    Automatically select image or PDF OCR based on file type.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    extension = os.path.splitext(file_path)[1].lower()

    image_extensions = {
        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
    }

    if extension in image_extensions:
        return extract_text_from_image(file_path)

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)

    raise ValueError(
        f"Unsupported file format: {extension}. "
        "Use an image or PDF file."
    )