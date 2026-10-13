
import json
from pathlib import Path

model_dir = Path("models")

files = [
    "tensorflow_class_names.json",
    "tensorflow_class_names_improved.json",
]

for filename in files:
    path = model_dir / filename

    print(f"\n{filename}")

    if not path.exists():
        print("File not found")
        continue

    with open(path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    print("Saved class order:", class_names)