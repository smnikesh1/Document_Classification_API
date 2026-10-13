
from pathlib import Path
from PIL import Image, ImageChops
import hashlib

folders = ["aadhar", "driving_license", "pan", "unknown"]

seen = {}

for folder in folders:
    for file_path in sorted((Path("dataset") / folder).glob("*.png")):
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()

        if digest in seen:
            print(f"Duplicate: {file_path}")
            print(f"  Same as: {seen[digest]}")
        else:
            seen[digest] = file_path

print(f"\nUnique images: {len(seen)}")
print("Check complete.") 