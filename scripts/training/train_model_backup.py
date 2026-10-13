
from pathlib import Path

import joblib
import pandas as pd
import pytesseract

from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Change this path if Tesseract is installed elsewhere.
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

CLASSES = [
    "aadhar",
    "pan",
    "driving_license",
]

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

MIN_IMAGES_PER_CLASS = 30
TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# 2. CLEAN OCR TEXT
# --------------------------------------------------

def clean_text(text):
    """Normalize OCR text for model training."""
    if not text:
        return ""

    return " ".join(text.split()).strip().lower()


# --------------------------------------------------
# 3. READ IMAGES AND EXTRACT TEXT
# --------------------------------------------------

def load_dataset():
    """Load document images and extract their OCR text."""

    texts = []
    labels = []
    skipped_files = []

    print("\n========== DATASET LOADING ==========\n")

    for document_type in CLASSES:
        folder = DATASET_DIR / document_type

        if not folder.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        readable_count = 0

        print(f"Reading folder: {document_type}")

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image = ImageOps.exif_transpose(image)
                    image = image.convert("RGB")
                    extracted_text = pytesseract.image_to_string(
                        image
                    )

                extracted_text = clean_text(extracted_text)

                if not extracted_text:
                    skipped_files.append(
                        (str(image_path), "No text detected")
                    )
                    print(f"  SKIPPED (no OCR text): {image_path.name}")
                    continue

                texts.append(extracted_text)
                labels.append(document_type)
                readable_count += 1

                print(f"  OK: {image_path.name}")

            except Exception as error:
                skipped_files.append(
                    (str(image_path), str(error))
                )
                print(f"  ERROR: {image_path.name} - {error}")

        print(
            f"Readable images for {document_type}: "
            f"{readable_count}"
        )

        if readable_count < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"\nNot enough readable images in '{folder}'. "
                f"Found {readable_count}; "
                f"need at least {MIN_IMAGES_PER_CLASS}.\n"
                "Add more varied, readable images and run again."
            )

    print("\n========== DATASET SUMMARY ==========\n")

    counts = pd.Series(labels).value_counts()

    for document_type in CLASSES:
        print(f"{document_type}: {counts.get(document_type, 0)}")

    print(f"\nTotal readable images: {len(texts)}")

    if skipped_files:
        print(f"Files skipped: {len(skipped_files)}")

        skipped_df = pd.DataFrame(
            skipped_files,
            columns=["File", "Reason"],
        )
        skipped_df.to_csv(
            MODEL_DIR / "skipped_training_files.csv",
            index=False,
        )

    return texts, labels


# --------------------------------------------------
# 4. TRAIN THE MODEL
# --------------------------------------------------

def train_model(texts, labels):
    """Split the data, vectorize OCR text, and train XGBoost."""

    print("\n========== SPLITTING DATA ==========\n")

    X_train_text, X_test_text, y_train, y_test = (
        train_test_split(
            texts,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
    )

    print(f"Training samples: {len(X_train_text)}")
    print(f"Testing samples: {len(X_test_text)}")

    print("\n========== TEXT VECTORIZATION ==========\n")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    # XGBoost expects integer labels.
    label_to_id = {
        document_type: index
        for index, document_type in enumerate(CLASSES)
    }

    id_to_label = {
        index: document_type
        for document_type, index in label_to_id.items()
    }

    y_train_encoded = [label_to_id[label] for label in y_train]
    y_test_encoded = [label_to_id[label] for label in y_test]

    print(f"TF-IDF training matrix: {X_train.shape}")
    print(f"TF-IDF testing matrix: {X_test.shape}")

    print("\n========== XGBOOST TRAINING ==========\n")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    model.fit(X_train, y_train_encoded)

    print("Model training completed.")

    # --------------------------------------------------
    # 5. EVALUATE THE MODEL
    # --------------------------------------------------

    print("\n========== MODEL EVALUATION ==========\n")

    predictions = model.predict(X_test).astype(int).tolist()

    accuracy = accuracy_score(y_test_encoded, predictions)

    print(f"Holdout accuracy: {accuracy:.2%}\n")

    print(
        classification_report(
            y_test_encoded,
            predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_test_encoded,
        predictions,
        labels=list(range(len(CLASSES))),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    matrix_path = MODEL_DIR / "xgboost_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    # Save per-image evaluation results.
    predicted_labels = [
        id_to_label[prediction]
        for prediction in predictions
    ]

    evaluation_df = pd.DataFrame({
        "Actual Document Type": y_test,
        "Model Prediction": predicted_labels,
        "Correct": [
            actual == predicted
            for actual, predicted in zip(
                y_test, predicted_labels
            )
        ],
    })

    evaluation_path = MODEL_DIR / "xgboost_evaluation.csv"
    evaluation_df.to_csv(evaluation_path, index=False)

    # --------------------------------------------------
    # 6. SAVE MODEL AND VECTORIZER
    # --------------------------------------------------

    model_path = MODEL_DIR / "xgboost_document_model.joblib"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    labels_path = MODEL_DIR / "document_labels.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(id_to_label, labels_path)

    print("\n========== FILES SAVED ==========\n")
    print(f"Model: {model_path}")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"Labels: {labels_path}")
    print(f"Evaluation report: {evaluation_path}")
    print(f"Confusion matrix: {matrix_path}")

    print("\nTraining completed successfully.")


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():
    texts, labels = load_dataset()
    train_model(texts, labels)


if __name__ == "__main__":
    main()