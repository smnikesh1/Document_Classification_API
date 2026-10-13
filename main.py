
from pathlib import Path
import shutil
import uuid

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException

app = FastAPI(title="Document Upload API")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

CLASSIFICATION_URL = "http://127.0.0.1:8001/classify"


@app.get("/health")
def health():
    return {"status": "healthy", "service": "upload"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    extension = ALLOWED_TYPES.get(file.content_type)

    if extension is None:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image.",
        )

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{extension}"

    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        response = requests.post(
            CLASSIFICATION_URL,
            json={"image_path": str(file_path.resolve())},
            timeout=120,
        )
        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        print(f"Classification service error: {error}")
        raise HTTPException(
            status_code=502,
            detail="The classification service is unavailable.",
        )
    finally:
        file_path.unlink(missing_ok=True)
        await file.close()
