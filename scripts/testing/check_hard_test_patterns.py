
from pathlib import Path
import re

from app.ocr import extract_text_from_image

folder = Path("dataset/hard_test")

patterns = {
    "aadhaar": r"\b(aadhaar|aadhar)\b",
    "pan": r"\bpan\b|\bpermanent account number\b",
    "driving_licence": r"\bdriving licence\b|\bdriving license\b",
    "transport_department": r"\btransport department\b",
    "date_of_birth": r"\bdate of birth\b",
}

for image_path in sorted(folder.iterdir()):
    if image_path.suffix.lower() not in {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }:
        continue

    text = extract_text_from_image(str(image_path)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = " ".join(text.split())

    found = [
        name for name, pattern in patterns.items()
        if re.search(pattern, text)
    ]

    print(f"{image_path.name}: {', '.join(found) if found else 'No keywords'}")