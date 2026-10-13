
from pathlib import Path
import json
import tensorflow as tf
import numpy as np
from PIL import Image

# Project folders
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "tensorflow_document_model_v5.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "tensorflow_class_names_v5.json"
TEST_DATASET_PATH = PROJECT_ROOT / "test_dataset"

IMAGE_SIZE = (224, 224)


def load_class_names():
    """Load the class names saved during training."""
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Support either a list or a dictionary in the JSON file.
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "class_names" in data:
            return data["class_names"]

        # Handle {"0": "aadhar", "1": "pan", ...}
        try:
            return [
                data[key]
                for key in sorted(data, key=lambda item: int(item))
            ]
        except (ValueError, TypeError):
            pass

    raise ValueError(
        "Could not read class names from the class-names JSON file."
    )


def predict_image(model, image_path, class_names):
    """Predict the top class without applying the confidence threshold."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.asarray(image, dtype=np.float32)

    # Add a batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    if predicted_index >= len(class_names):
        raise ValueError(
            "The model output does not match the number of class names."
        )

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


def main():
    # Check required files
    if not MODEL_PATH.exists():
        print(f"ERROR: Model file not found:\n{MODEL_PATH}")
        return

    if not CLASS_NAMES_PATH.exists():
        print(f"ERROR: Class names file not found:\n{CLASS_NAMES_PATH}")
        return

    if not TEST_DATASET_PATH.exists():
        print(f"ERROR: Test dataset folder not found:\n{TEST_DATASET_PATH}")
        return

    print("Loading v5 model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_names = load_class_names()

    print("Model loaded successfully.")
    print(f"Classes: {class_names}")
    print("\nStarting evaluation on unseen test images...\n")

    total_images = 0
    correct_predictions = 0
    category_results = {}

    for category in class_names:
        category_folder = TEST_DATASET_PATH / category

        if not category_folder.exists():
            print(f"WARNING: Folder not found: {category_folder}")
            continue

        image_files = [
            path
            for path in category_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        category_total = 0
        category_correct = 0

        print(f"\n--- Actual category: {category} ---")

        if not image_files:
            print("No supported images found in this folder.")
            continue

        for image_path in sorted(image_files):
            try:
                predicted_class, confidence = predict_image(
                    model, image_path, class_names
                )

                is_correct = predicted_class.lower() == category.lower()

                total_images += 1
                category_total += 1

                if is_correct:
                    correct_predictions += 1
                    category_correct += 1

                result = "CORRECT" if is_correct else "WRONG"

                print(
                    f"{image_path.name} | "
                    f"Predicted: {predicted_class} | "
                    f"Confidence: {confidence:.2f}% | "
                    f"{result}"
                )

            except Exception as error:
                print(f"ERROR processing {image_path.name}: {error}")

        category_results[category] = (
            category_correct,
            category_total
        )

    print("\n" + "=" * 50)
    print("V5 MODEL EVALUATION SUMMARY")
    print("=" * 50)

    for category, (correct, total) in category_results.items():
        accuracy = (correct / total * 100) if total else 0
        print(
            f"{category}: {correct}/{total} correct "
            f"({accuracy:.2f}%)"
        )

    overall_accuracy = (
        correct_predictions / total_images * 100
        if total_images
        else 0
    )

    print("-" * 50)
    print(
        f"Overall raw top-1 accuracy: "
        f"{correct_predictions}/{total_images} "
        f"({overall_accuracy:.2f}%)"
    )
    print("=" * 50)
    print(
        "\nNote: This test reports the model's top prediction "
        "without the API's 60% confidence threshold."
    )
    print(
        "It is for diagnosis only; it does not change your model "
        "or API."
    )


if __name__ == "__main__":
    main()

