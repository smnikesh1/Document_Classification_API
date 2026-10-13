
from pathlib import Path
import numpy as np
import tensorflow as tf

DATASET_DIR = Path("dataset")
MODEL_PATH = "models/tensorflow_document_model.keras"

CLASS_NAMES = ["aadhar", "driving_license", "pan", "unknown"]

# Load a validation split with stable file order.
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=1,
    shuffle=False,
)

model = tf.keras.models.load_model(MODEL_PATH)

print("\nPAN VALIDATION ERRORS")
print("-" * 50)

errors_found = 0

# Dataset batches and file paths now use the same order.
for index, (images, labels) in enumerate(validation_data):
    true_label = int(labels.numpy()[0])
    probabilities = model.predict(images, verbose=0)[0]
    predicted_label = int(np.argmax(probabilities))

    if true_label == 2 and predicted_label != 2:
        filename = Path(validation_data.file_paths[index]).name

        print(f"File: {filename}")
        print(f"Expected: {CLASS_NAMES[true_label]}")
        print(f"Predicted: {CLASS_NAMES[predicted_label]}")
        print(f"Confidence: {probabilities[predicted_label] * 100:.2f}%")
        print("-" * 30)

        errors_found += 1

if errors_found == 0:
    print("No PAN validation errors found in this split.")

print(f"\nTotal PAN validation errors: {errors_found}")
print("Inspection complete.")