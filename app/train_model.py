
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from app.ocr import extract_text
from app.cleaner import clean_text


# ==========================================
# 1. SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

CLASSES = ["aadhar", "pan", "driving_license"]

MAX_TOKENS = 5000
SEQUENCE_LENGTH = 200
BATCH_SIZE = 4
EPOCHS = 20
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. LOAD OCR TEXT AND LABELS
# ==========================================

def load_dataset():
    texts = []
    labels = []

    for label in CLASSES:
        folder = DATASET_DIR / label

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Dataset folder not found: {folder}"
            )

        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in {
                ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
            }:
                continue

            print(f"Reading: {image_path.relative_to(BASE_DIR)}")

            try:
                text = clean_text(extract_text(str(image_path)))
            except Exception as exc:
                print(f"Warning: Could not read {image_path.name}: {exc}")
                continue

            if not text.strip():
                print(f"Warning: No text found in {image_path.name}")
                continue

            texts.append(text)
            labels.append(CLASSES.index(label))

    return texts, labels


# ==========================================
# 3. SPLIT DATA WITHOUT SKLEARN
# ==========================================

def split_dataset(texts, labels):
    class_indices = {i: [] for i in range(len(CLASSES))}

    for index, label in enumerate(labels):
        class_indices[label].append(index)

    train_indices = []
    val_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)

        if len(indices) < 2:
            raise ValueError(
                f"Class '{CLASSES[label]}' needs at least two readable images."
            )

        val_count = max(1, round(len(indices) * 0.25))
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    X_train = [texts[i] for i in train_indices]
    y_train = [labels[i] for i in train_indices]
    X_val = [texts[i] for i in val_indices]
    y_val = [labels[i] for i in val_indices]

    return X_train, y_train, X_val, y_val


# ==========================================
# 4. BUILD TENSORFLOW TEXT MODEL
# ==========================================

def build_model(training_texts):
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
        standardize="lower_and_strip_punctuation",
    )

    text_dataset = tf.data.Dataset.from_tensor_slices(
        training_texts
    ).batch(BATCH_SIZE)

    vectorizer.adapt(text_dataset)

    inputs = tf.keras.Input(
        shape=(),
        dtype=tf.string,
        name="ocr_text",
    )

    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(
        input_dim=MAX_TOKENS,
        output_dim=64,
        mask_zero=True,
    )(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(CLASSES),
        activation="softmax",
    )(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ==========================================
# 5. TRAIN, EVALUATE, AND SAVE
# ==========================================

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    texts, labels = load_dataset()

    if len(texts) < 12 or len(set(labels)) != len(CLASSES):
        raise ValueError(
            "Not enough readable data. Each class needs readable images."
        )

    print(f"\nTotal readable images: {len(texts)}")
    print("Class counts:")
    print(pd.Series([CLASSES[i] for i in labels]).value_counts())

    X_train, y_train, X_val, y_val = split_dataset(texts, labels)

    model = build_model(X_train)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (X_train, np.asarray(y_train, dtype=np.int32))
    ).shuffle(len(X_train), seed=SEED).batch(BATCH_SIZE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (X_val, np.asarray(y_val, dtype=np.int32))
    ).batch(BATCH_SIZE)

    print("\nTraining TensorFlow OCR text classifier...")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
            )
        ],
    )

    probabilities = model.predict(val_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    actual = np.asarray(y_val)

    accuracy = float(np.mean(predictions == actual))
    print(f"\nValidation accuracy: {accuracy:.2%}")

    # Per-class evaluation without sklearn.
    for class_index, class_name in enumerate(CLASSES):
        true_positive = int(np.sum(
            (actual == class_index) & (predictions == class_index)
        ))
        false_positive = int(np.sum(
            (actual != class_index) & (predictions == class_index)
        ))
        false_negative = int(np.sum(
            (actual == class_index) & (predictions != class_index)
        ))

        precision = true_positive / max(
            true_positive + false_positive, 1
        )
        recall = true_positive / max(
            true_positive + false_negative, 1
        )

        print(
            f"{class_name}: precision={precision:.2f}, "
            f"recall={recall:.2f}"
        )

    model_path = MODEL_DIR / "tensorflow_ocr_text_model.keras"
    model.save(model_path)

    report_path = MODEL_DIR / "tensorflow_ocr_evaluation.csv"
    pd.DataFrame({
        "actual": [CLASSES[i] for i in actual],
        "predicted": [CLASSES[i] for i in predictions],
        "correct": predictions == actual,
    }).to_csv(report_path, index=False)

    with open(
        MODEL_DIR / "tensorflow_ocr_class_names.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(CLASSES, file, indent=2)

    print(f"\nSaved TensorFlow model: {model_path}")
    print(f"Saved evaluation: {report_path}")
    print("Training complete.")


if __name__ == "__main__":
    main()