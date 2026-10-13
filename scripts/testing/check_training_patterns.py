
from pathlib import Path
from collections import Counter
import re

from app.ocr import extract_text_from_image

dataset = Path("dataset")
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if not dataset.exists():
    raise SystemExit("ERROR: dataset folder not found.")

for folder in sorted(dataset.iterdir()):
    if not folder.is_dir() or folder.name == "hard_test":
        continue

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    counts = Counter()

    for image_path in images:
        text = extract_text_from_image(str(image_path)).lower()

        # Treat hyphens and punctuation like spaces
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = " ".join(text.split())

        patterns = {
            "aadhaar": r"\b(aadhaar|aadhar)\b",
            "pan": r"\bpan\b|\bpermanent account number\b",
            "driving_licence": r"\bdriving licence\b|\bdriving license\b",
            "transport_department": r"\btransport department\b",
            "date_of_birth": r"\bdate of birth\b",
        }

        for name, pattern in patterns.items():
            if re.search(pattern, text):
                counts[name] += 1

    print(f"\nClass: {folder.name}")
    print(f"Total images: {len(images)}")

    for name in patterns:
        print(f"{name}: {counts[name]}/{len(images)}")