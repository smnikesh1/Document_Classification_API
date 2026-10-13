
from pathlib import Path

import joblib
import pandas as pd
import pytesseract

from PIL import Image, ImageOps, ImageEnhance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Four document categories
CLASSES = [
    "aadhar",
    "pan",
    "driving_license",
    "unknown",
]

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg",
    ".bmp", ".tif", ".tiff",
}

# Five is a temporary minimum to allow initial testing.
# It is not enough to build a reliable real-world classifier.
MIN_IMAGES_PER_CLASS = 5

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==================================================
# 2. CLEAN OCR TEXT
# ==================================================

def clean_text(text):
    """Normalize OCR text before model training."""

    if not text:
        return ""

    return " ".join(text.split()).strip().lower()


# ==================================================
# 3. EXTRACT OCR TEXT FROM IMAGES
# ==================================================

def extract_text_from_image(image_path):
    """Read an image and extract text using Tesseract."""

    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source)
        image = image.convert("RGB")

        # Create a grayscale copy to help OCR read tex
        gray = ImageOps.grayscale(image)
        gray = ImageEnhance.Contrast(gray).enhance(1.5)

        text = pytesseract.image_to_string(
            gray,
            config="--psm 6",
        )

    return clean_text(text)


# ==================================================
# 4. LOAD THE FOUR TRAINING CLASSES
# ==================================================

def load_dataset():
    """
    Load only the four class folders.
    The hard_test folder is never used for training.
    """

    texts = []
    labels = []
    skipped_files = []

    print("\n========== DATASET LOADING ==========\n")

    for document_type in CLASSES:
        folder = DATASET_DIR / document_type

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing class folder: {folder}"
            )

        image_paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        print(f"\nReading class: {document_type}")
        readable_count = 0

        for image_path in image_paths:
            try:
                extracted_text = extract_text_from_image(
                    image_path
                )

                if not extracted_text:
                    skipped_files.append({
                        "File": str(image_path),
                        "Class": document_type,
                        "Reason": "No OCR text detected",
                    })
                    print(f"  SKIPPED: {image_path.name}")
                    continue

                texts.append(extracted_text)
                labels.append(document_type)
                readable_count += 1

                print(f"  OK: {image_path.name}")

            except Exception as error:
                skipped_files.append({
                    "File": str(image_path),
                    "Class": document_type,
                    "Reason": str(error),
                })
                print(f"  ERROR: {image_path.name}: {error}")

        print(
            f"Readable images for {document_type}: "
            f"{readable_count}"
        )

        if readable_count < MIN_IMAGES_PER_CLASS:
            raise ValueError(
                f"Class '{document_type}' has only "
                f"{readable_count} readable images. "
                f"At least {MIN_IMAGES_PER_CLASS} are required "
                "for this initial experiment."
            )

    # Save information about unreadable images.
    skipped_path = MODEL_DIR / "skipped_training_files.csv"

    pd.DataFrame(
        skipped_files,
        columns=["File", "Class", "Reason"],
    ).to_csv(skipped_path, index=False)

    counts = pd.Series(labels).value_counts()

    print("\n========== DATASET SUMMARY ==========\n")

    for document_type in CLASSES:
        print(
            f"{document_type}: "
            f"{counts.get(document_type, 0)} readable images"
        )

    print(f"Total readable images: {len(texts)}")
    print(f"Skipped-file report: {skipped_path}")

    return texts, labels


# ==================================================
# 5. TRAIN AND EVALUATE THE MODEL
# ==================================================

def train_model(texts, labels):
    """Split the data, train XGBoost, and evaluate it."""

    print("\n========== DATA SPLIT ==========\n")

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
    print(f"Internal holdout samples: {len(X_test_text)}")

    print("\n========== TEXT VECTORIZATION ==========\n")

    # Fit only on training text to avoid data leakage.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(X_train_text) #converts training text into numerical features.
    X_test = vectorizer.transform(X_test_text)

    print(f"Training matrix: {X_train.shape}")
    print(f"Testing matrix: {X_test.shape}")

    # Map category names to numeric IDs.
    label_to_id = {
        name: index
        for index, name in enumerate(CLASSES)
    }

    id_to_label = {
        index: name
        for name, index in label_to_id.items()
    }

    y_train_encoded = [
        label_to_id[label] for label in y_train
    ]

    y_test_encoded = [
        label_to_id[label] for label in y_test
    ]

    print("\n========== XGBOOST TRAINING ==========\n")

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASSES),
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )

    model.fit(X_train, y_train_encoded)

    print("Training completed.")

    # ----------------------------------------------
    # Evaluate on the internal holdout split.
    # ----------------------------------------------

    print("\n========== INTERNAL EVALUATION ==========\n")

    predicted_ids = model.predict(X_test).astype(int).tolist()

    predicted_labels = [
        id_to_label[index] for index in predicted_ids
    ]

    accuracy = accuracy_score(
        y_test,
        predicted_labels,
    )

    print(f"Internal holdout accuracy: {accuracy:.2%}")
    print(
        "\nThis result is from the internal split, "
        "not the separate hard_test dataset.\n"
    )

    print(
        classification_report(
            y_test,
            predicted_labels,
            labels=CLASSES,
            target_names=CLASSES,
            zero_division=0,
        )
    )

    # Save confusion matrix.
    matrix = confusion_matrix(
        y_test,
        predicted_labels,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    matrix_path = MODEL_DIR / "xgboost_confusion_matrix.csv"
    matrix_df.to_csv(matrix_path)

    # Save predictions for each internal holdout sample.
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

    # ----------------------------------------------
    # Save model artifacts.
    # ----------------------------------------------

    model_path = MODEL_DIR / "xgboost_document_model.joblib"
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    labels_path = MODEL_DIR / "document_labels.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(id_to_label, labels_path)

    # Save a summary of the trained classes.
    summary_df = pd.DataFrame({
        "Document Type": CLASSES,
        "Training Samples": [
            sum(
                label == document_type
                for label in y_train
            )
            for document_type in CLASSES
        ],
        "Internal Holdout Samples": [
            sum(
                label == document_type
                for label in y_test
            )
            for document_type in CLASSES
        ],
    })

    summary_path = MODEL_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n========== FILES SAVED ==========\n")
    print(f"Model: {model_path}")
    print(f"Vectorizer: {vectorizer_path}")
    print(f"Labels: {labels_path}")
    print(f"Evaluation: {evaluation_path}")
    print(f"Confusion matrix: {matrix_path}")
    print(f"Training summary: {summary_path}")

    print("\nTraining completed successfully.")


# ==================================================
# 6. MAIN PROGRAM
# ==================================================

def main():
    print("FOUR-CLASS DOCUMENT CLASSIFICATION")
    print("Classes:", ", ".join(CLASSES))
    print("Hard-test images are excluded from training.")

    texts, labels = load_dataset()
    train_model(texts, labels)


if __name__ == "__main__":
    main()