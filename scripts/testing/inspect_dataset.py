
from pathlib import Path
from PIL import Image

folders = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
    "hard_test",
]

total = 0
bad_files = []

for folder in folders:
    folder_path = Path("dataset") / folder

    for file_path in folder_path.iterdir():
        if not file_path.is_file():
            continue

        total += 1

        try:
            with Image.open(file_path) as image:
                image.verify()
        except Exception as error:
            bad_files.append(file_path)
            print(f"Problem: {file_path} — {error}")

print("\n--- Dataset Image Check ---")
print(f"Total files checked: {total}")
print(f"Unreadable or corrupted files: {len(bad_files)}")

if not bad_files:
    print("All image files passed the file-integrity check.")