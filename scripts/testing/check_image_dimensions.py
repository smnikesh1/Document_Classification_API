
from pathlib import Path
from collections import Counter
from PIL import Image

folders = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
    "hard_test",
]

sizes = Counter()
formats = Counter()

for folder in folders:
    for file_path in (Path("dataset") / folder).iterdir():
        if not file_path.is_file():
            continue

        try:
            with Image.open(file_path) as image:
                sizes[image.size] += 1
                formats[image.format] += 1
        except Exception:
            pass

print("Image dimensions:")
for size, count in sizes.most_common():
    print(f"{size}: {count} images")

print("\nImage formats:")
for image_format, count in formats.items():
    print(f"{image_format}: {count} images")