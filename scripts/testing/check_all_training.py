
from pathlib import Path
from collections import Counter
from app.ocr import extract_text_from_image

dataset = Path("dataset")
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if not dataset.exists():
    raise SystemExit("ERROR: dataset folder not found.")

for class_folder in sorted(dataset.iterdir()):
    if not class_folder.is_dir():
        continue

    images = sorted(
        p for p in class_folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        continue

    counts = Counter()

    for image_path in images:
        text = " ".join(
            extract_text_from_image(str(image_path)).lower().split()
        )

        if "driving licence" in text or "driving license" in text:
            counts["driving_licence_term"] += 1

        if "transport department" in text:
            counts["transport_department_term"] += 1

        if "synthetic test document" in text:
            counts["synthetic_test_document_term"] += 1

        if "date of birth" in text:
            counts["date_of_birth_term"] += 1

        if "permanent account number" in text:
            counts["pan_term"] += 1

        if "aadhaar" in text or "aadhar" in text:
            counts["aadhaar_term"] += 1

    print("\nClass:", class_folder.name)
    print("Total images:", len(images))

    for term, count in sorted(counts.items()):
        print(f"{term}: {count}/{len(images)}")