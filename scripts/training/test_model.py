
import sys
from pathlib import Path

# --------------------------------------------------
# 1. Find the main project directory
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# Add the project directory so Python can find app
sys.path.insert(0, str(BASE_DIR))

# --------------------------------------------------
# 2. Import the image classification function
# --------------------------------------------------

from app.image_classifier import classify_image

# --------------------------------------------------
# 3. Dataset configuration
# --------------------------------------------------

DATASET_DIR = BASE_DIR / "dataset"

CATEGORIES = [
    "aadhar",
    "pan",
    "driving_license",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# --------------------------------------------------
# 4. Track test results
# --------------------------------------------------

total = 0
correct = 0
incorrect = 0
failed = 0

results = []

# --------------------------------------------------
# 5. Test images in each category
# --------------------------------------------------

print("=" * 60)
print("TENSORFLOW DOCUMENT CLASSIFICATION TEST")
print("=" * 60)

for actual_class in CATEGORIES:

    class_dir = DATASET_DIR / actual_class

    if not class_dir.exists():
        print(f"\nMissing category folder: {class_dir}")
        continue

    image_paths = [
        path
        for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    print(f"\nTesting category: {actual_class}")
    print(f"Images found: {len(image_paths)}")
    print("-" * 60)

    for image_path in image_paths:

        try:
            # Predict the document category
            prediction = classify_image(str(image_path))

            predicted_class = prediction["classified"]
            confidence = prediction["confidence"]

            # Compare prediction with the actual folder
            is_correct = predicted_class == actual_class

            total += 1

            if is_correct:
                correct += 1
                status = "CORRECT"
            else:
                incorrect += 1
                status = "WRONG"

            # Save the result
            results.append({
                "image": image_path.name,
                "actual": actual_class,
                "predicted": predicted_class,
                "confidence": confidence,
                "correct": is_correct,
            })

            # Print individual result
            print(
                f"Image: {image_path.name} | "
                f"Actual: {actual_class} | "
                f"Predicted: {predicted_class} | "
                f"Confidence: {confidence:.2%} | "
                f"{status}"
            )

        except Exception as error:
            failed += 1
            print(
                f"ERROR testing {image_path.name}: {error}"
            )

# --------------------------------------------------
# 6. Overall test summary
# --------------------------------------------------

print("\n")
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(f"Images tested: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {incorrect}")
print(f"Failed tests: {failed}")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"Accuracy on tested images: {accuracy:.2f}%")
else:
    print("No images were successfully tested.")

# --------------------------------------------------
# 7. Results for each category
# --------------------------------------------------

print("\n")
print("=" * 60)
print("RESULTS BY CATEGORY")
print("=" * 60)

for category in CATEGORIES:

    category_results = [
        result
        for result in results
        if result["actual"] == category
    ]

    category_correct = sum(
        result["correct"]
        for result in category_results
    )

    category_total = len(category_results)

    if category_total > 0:
        category_accuracy = (
            category_correct / category_total
        ) * 100

        print(
            f"{category}: "
            f"{category_correct}/{category_total} correct "
            f"({category_accuracy:.2f}%)"
        )
    else:
        print(f"{category}: No successful tests")

print("\nTesting completed.")