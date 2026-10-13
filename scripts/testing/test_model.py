
from pathlib import Path
from app.ocr import extract_text_from_image

# Folder containing driving-licence training images
folder = Path("dataset/driving_license")

if not folder.exists():
    print("ERROR: Folder not found:", folder)
    raise SystemExit(1)

image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
images = sorted(
    file for file in folder.iterdir()
    if file.is_file() and file.suffix.lower() in image_extensions
)

print("Total images found:", len(images))

for image_path in images:
    text = extract_text_from_image(str(image_path))
    normalized_text = " ".join(text.lower().split())

    has_synthetic = "synthetic test document" in normalized_text
    has_dob = "date of birth" in normalized_text
    has_driving = (
        "driving licence" in normalized_text
        or "driving license" in normalized_text
    )
    has_transport = "transport department" in normalized_text

    print("\nImage:", image_path.name)
    print("  Driving licence wording:", has_driving)
    print("  Transport department:", has_transport)
    print("  Synthetic test document:", has_synthetic)
    print("  Date of birth:", has_dob)
    print("  OCR text:", text[:250])