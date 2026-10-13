
from pathlib import Path
import tensorflow as tf

DATASET_DIR = Path("dataset")
MODEL_PATH = Path("models/tensorflow_document_model.keras")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

CLASS_NAMES = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]

# Re-create the original validation split.
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

model = tf.keras.models.load_model(MODEL_PATH)

correct = 0
total = 0
class_results = {
    name: {"correct": 0, "total": 0}
    for name in CLASS_NAMES
}

print("\nVALIDATION PREDICTIONS")
print("-" * 65)

for images, labels in validation_data:
    predictions = model.predict(images, verbose=0)

    for i, scores in enumerate(predictions):
        expected = CLASS_NAMES[int(labels[i])]
        predicted = CLASS_NAMES[int(scores.argmax())]
        confidence = float(scores.max()) * 100

        is_correct = expected == predicted
        correct += int(is_correct)
        total += 1

        class_results[expected]["total"] += 1
        class_results[expected]["correct"] += int(is_correct)

        status = "CORRECT" if is_correct else "WRONG"

        print(
            f"Expected: {expected:16} | "
            f"Predicted: {predicted:16} | "
            f"Confidence: {confidence:5.2f}% | {status}"
        )

print("\nRESULTS BY CLASS")
print("-" * 40)

for name, result in class_results.items():
    count = result["total"]
    good = result["correct"]
    accuracy = good / count * 100 if count else 0

    print(
        f"{name:16}: {good}/{count} correct "
        f"({accuracy:.2f}%)"
    )

print("-" * 40)
print(f"Overall: {correct}/{total} correct")
print(f"Accuracy: {correct / total * 100:.2f}%")