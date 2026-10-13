
import easyocr
from pathlib import Path

# Create the OCR reader
reader = easyocr.Reader(["en"], gpu=False)

# Folder containing the 20 test images
image_folder = Path("dataset/hard_test")

# Read every PNG image
for image_path in sorted(image_folder.glob("*.png")):
    print("\n" + "=" * 50)
    print(f"IMAGE: {image_path.name}")

    try:
        text = reader.readtext(str(image_path), detail=0)

        if text:
            for line in text:
                print(line)
        else:
            print("No text detected.")

    except Exception as error:
        print(f"Error reading image: {error}")