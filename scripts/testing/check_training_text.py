
from pathlib import Path
from app.ocr import extract_text_from_image

folder = Path("dataset/driving_license")

if not folder.exists():
    print("ERROR: Folder not found:", folder)
    raise SystemExit(1)

extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

images = sorted(
    file for file in folder.iterdir()
    if file.is_file() and file.suffix.lower() in extensions
)

print("Total images found:", len(images))

for image_path in images:
    text = extract_text_from_image(str(image_path))
    normalized = " ".join(text.lower().split())

    print("\nImage:", image_path.name)
    print("Driving licence:",
          "driving licence" in normalized or
          "driving license" in normalized)
    print("Transport department:",
          "transport department" in normalized)
    print("Synthetic test document:",
          "synthetic test document" in normalized)
    print("Date of birth:",
          "date of birth" in normalized)
    print("OCR text:", text[:250])