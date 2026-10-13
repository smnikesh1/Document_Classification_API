import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "tensorflow_document_model_v5.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "tensorflow_class_names_v5.json"

# --------------------------------------------------
# 2. Configuration
# --------------------------------------------------

IMAGE_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60

EXPECTED_CLASSES = {
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
}

# --------------------------------------------------
# 3. Check required files
# --------------------------------------------------

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"TensorFlow v5 model not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.is_file():
    raise FileNotFoundError(f"Class names file not found: {CLASS_NAMES_PATH}")

# --------------------------------------------------
# 4. Load model
# --------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
)

# --------------------------------------------------
# 5. Load and validate class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

if not isinstance(CLASS_NAMES, list):
    raise ValueError("Class names must be stored as a JSON list.")

if len(CLASS_NAMES) != 4:
    raise ValueError(f"Expected four classes, got: {CLASS_NAMES}")

if set(CLASS_NAMES) != EXPECTED_CLASSES:
    raise ValueError(f"Unexpected class names: {CLASS_NAMES}")

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError("Model output count does not match class names.")

print("TensorFlow v5 document classification model loaded successfully.")
print("Model path:", MODEL_PATH)
print("Categories:", CLASS_NAMES)

# --------------------------------------------------
# Helper: Run Inference on a PIL Image
# --------------------------------------------------

def _predict_pil_image(image: Image.Image) -> dict:
    image = image.convert("RGB")

    # Maintain original aspect ratio: scale to fit inside 224x224 and pad neutral gray borders
    image.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS)
    delta_w = IMAGE_SIZE[0] - image.size[0]
    delta_h = IMAGE_SIZE[1] - image.size[1]
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    padded_image = ImageOps.expand(image, padding, fill=(128, 128, 128))

    # Keep pixel values in the 0-255 range (MobileNet preprocessing is internal)
    image_array = np.asarray(padded_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    if (
        len(predictions) != len(CLASS_NAMES)
        or not np.all(np.isfinite(predictions))
        or np.any(predictions < 0)
        or np.any(predictions > 1)
    ):
        raise ValueError("The model returned invalid prediction scores.")

    if not np.isclose(np.sum(predictions), 1.0, atol=0.01):
        raise ValueError("The model returned invalid probability scores.")

    # Calculate probabilities across all classes
    all_scores = {
        CLASS_NAMES[i]: round(float(predictions[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    best_index = int(np.argmax(predictions))
    predicted_category = CLASS_NAMES[best_index]
    confidence = round(float(predictions[best_index]), 4)

    status = (
        predicted_category
        if confidence >= CONFIDENCE_THRESHOLD
        else "uncertain"
    )

    return {
        "classified": status,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "all_scores": all_scores,
        "message": "Prediction completed successfully.",
    }

# --------------------------------------------------
# 6. Classification Functions
# --------------------------------------------------

def classify_image_bytes(image_bytes: bytes) -> dict:
    """
    Classify an uploaded image directly from memory bytes.
    Used by FastAPI endpoints.
    """
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return _predict_pil_image(img)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Please provide a valid image file.") from exc


def classify_image(image_path: str) -> dict:
    """
    Classify an image given a file path on disk.
    """
    try:
        with Image.open(image_path) as img:
            return _predict_pil_image(img)
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Please provide a valid image file.") from exc