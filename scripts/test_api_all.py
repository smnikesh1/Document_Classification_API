
from pathlib import Path
import requests

API_URL = "http://127.0.0.1:8000/classify"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DATASET = PROJECT_ROOT / "test_dataset"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

total = 0
correct = 0
failed = 0

print("\nTesting all images through the API...\n")

for category in [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]:
    folder = TEST_DATASET / category

    if not folder.is_dir():
        print(f"Missing folder: {folder}")
        continue

    category_total = 0
    category_correct = 0

    for image_path in sorted(folder.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            with image_path.open("rb") as image_file:
                response = requests.post(
                    API_URL,
                    files={
                        "file": (
                            image_path.name,
                            image_file,
                            "image/png"
                            if image_path.suffix.lower() == ".png"
                            else "image/jpeg",
                        )
                    },
                    timeout=60,
                )

            total += 1
            category_total += 1

            if response.status_code != 200:
                failed += 1
                print(
                    f"{image_path.name}: API ERROR "
                    f"{response.status_code} - {response.text}"
                )
                continue

            result = response.json()
            predicted = result.get("classified", "missing")
            confidence = result.get("confidence", 0)

            is_correct = predicted == category

            if is_correct:
                correct += 1
                category_correct += 1

            status = "CORRECT" if is_correct else "CHECK"

            print(
                f"{image_path.name} | "
                f"Expected: {category} | "
                f"Predicted: {predicted} | "
                f"Confidence: {confidence} | {status}"
            )

        except requests.RequestException as error:
            failed += 1
            print(f"{image_path.name}: Request failed - {error}")

    if category_total:
        print(
            f"{category}: {category_correct}/{category_total} "
            "correct\n"
        )

print("=" * 45)
print("API TEST SUMMARY")
print("=" * 45)
print(f"Images tested: {total}")
print(f"Correct classifications: {correct}")
print(f"API/request failures: {failed}")

if total:
    print(f"API accuracy: {correct / total * 100:.2f}%")

print("=" * 45)
print("Note: 'uncertain' predictions count as incorrect.")

