
from pathlib import Path
from PIL import Image
import hashlib

folders = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]

seen = {}
duplicates = []

for folder in folders:
    folder_path = Path("dataset") / folder

    for file_path in sorted(folder_path.glob("*.png")):
        with Image.open(file_path) as image:
            image_hash = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()

        if image_hash in seen:
            duplicates.append((str(file_path), seen[image_hash]))
        else:
            seen[image_hash] = str(file_path)

print("\n--- Exact Duplicate Image Check ---")
print("Unique images:", len(seen))
print("Duplicate images:", len(duplicates))

for duplicate, original in duplicates:
    print(f"Duplicate: {duplicate}")
    print(f"Matches:   {original}")