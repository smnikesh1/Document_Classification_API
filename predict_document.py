
from pathlib import Path

import joblib
import pandas as pd
import pytesseract

from PIL import Image, ImageOps, ImageEnhance


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "xgboost_document_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
LABELS_PATH = MODEL_DIR / "document_labels.joblib"

TEST_DIR = DATASET_DIR / "hard_test"
OUTPUT_PATH = MODEL_DIR / "hard_test_predictions.csv"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


# ==================================================
# 2. CLEAN OCR TEXT
# ==================================================

def clean_text(text):
    """Normalize OCR text in the same way as training."""

    if not text:
        return ""

    return " ".join(text.split()).strip().lower()


# ==================================================
# 3. EXTRACT OCR TEXT
# ==================================================

def extract_text(image_path):
    """Apply the same image preprocessing used in training."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

        gray = ImageOps.grayscale(image)
        gray = ImageEnhance.Contrast(gray).enhance(1.5)

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6",
        )

    return clean_text(text)


# ==================================================
# 4. LOAD TRAINED FILES
# ==================================================

def load_model_files():
    required_files = [
        MODEL_PATH,
        VECTORIZER_PATH,
        LABELS_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model file: {path}\n"
                "Run python train_model.py first."
            )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    id_to_label = joblib.load(LABELS_PATH)

    print("XGBoost model loaded.")
    print("TF-IDF vectorizer loaded.")
    print("Document labels loaded.")
    print("Available classes:", list(id_to_label.values()))

    return model, vectorizer, id_to_label


# ==================================================
# 5. PREDICT ONE IMAGE
# ==================================================

def predict_one(image_path, model, vectorizer, id_to_label):
    """Return OCR text, predicted class, confidence and probabilities."""

    text = extract_text(image_path)

    if not text:
        return "", "unreadable", 0.0, {}

    features = vectorizer.transform([text])

    predicted_id = int(model.predict(features)[0])
    prediction = id_to_label[predicted_id]

    probabilities = model.predict_proba(features)[0]

    class_probabilities = {
        id_to_label[index]: round(
            float(probability) * 100, 2
        )
        for index, probability in enumerate(probabilities)
    }

    confidence = class_probabilities[prediction]

    return text, prediction, confidence, class_probabilities


# ==================================================
# 6. PROCESS HARD-TEST IMAGES
# ==================================================

def main():
    model, vectorizer, id_to_label = load_model_files()

    if not TEST_DIR.is_dir():
        raise FileNotFoundError(
            f"Test folder not found: {TEST_DIR}"
        )

    image_files = sorted(
        path
        for path in TEST_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"No images found in {TEST_DIR}")
        return

    rows = []

    print("\n========== HARD-TEST PREDICTIONS ==========\n")

    for number, image_path in enumerate(image_files, start=1):
        try:
            text, prediction, confidence, probabilities = (
                predict_one(
                    image_path,
                    model,
                    vectorizer,
                    id_to_label,
                )
            )

            rows.append({
                "SI No": number,
                "Document Name": image_path.name,
                "OCR Text": text,
                "Model Prediction": prediction,
                "Confidence": confidence,
                "Class Probabilities": str(probabilities),
            })

            print(f"Image: {image_path.name}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Class probabilities: {probabilities}")
            print("-" * 60)

        except Exception as error:
            print(f"Error processing {image_path.name}: {error}")

            rows.append({
                "SI No": number,
                "Document Name": image_path.name,
                "OCR Text": "",
                "Model Prediction": "error",
                "Confidence": 0.0,
                "Class Probabilities": "{}",
            })

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n========== COMPLETED ==========")
    print(f"Images found: {len(image_files)}")
    print(f"Results saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()