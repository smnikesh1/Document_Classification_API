
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

# Re-create the same validation split used during training.
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

# Load the original model without changing it.
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate validation accuracy and loss.
loss, accuracy = model.evaluate(validation_data, verbose=1)

print("\nORIGINAL MODEL - VALIDATION RESULTS")
print("-" * 40)
print(f"Validation images: {len(validation_data.file_paths)}")
print(f"Validation loss: {loss:.4f}")
print(f"Validation accuracy: {accuracy * 100:.2f}%")
print("-" * 40)