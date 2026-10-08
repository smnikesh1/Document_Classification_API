from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from app.ocr import extract_text
from app.cleaner import clean_text
from app.classifier import classify_document


app = FastAPI(title="Document Classification API")


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.post("/classify")
async def classify(file: UploadFile = File(...)):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "application/pdf"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, JPG and PNG files are allowed"
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extracted_text = extract_text(str(file_path))

    cleaned_text = clean_text(extracted_text)
    result = classify_document(cleaned_text)

    return {
        "classified": result["classified"],
        "confidence": result["confidence"]
    }