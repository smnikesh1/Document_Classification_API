
from pathlib import Path
import numpy as np
import tensorflow as tf

# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"

# 2. Model settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

CLASS_NAMES = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]


def main():
    print("\n" + "=" * 50)
    print("DOCUMENT CLASSIFICATION MODEL COMPARISON")
    print("=" * 50)

    # Check that the folders exist
    if not DATASET_DIR.exists():
        print("ERROR: Dataset folder was not found.")
        return

    for name in CLASS_NAMES:
        if not (DATASET_DIR / name).is_dir():
            print(f"ERROR: Missing dataset folder: {name}")
            return

    # Load only the four document classes.
    # hard_test is excluded.
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        class_names=CLASS_NAMES,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    print("\nClasses:", dataset.class_names)

    # Gather the actual labels
    y_true = np.concatenate(
        [labels.numpy() for _, labels in dataset]
    )

    # Gather the actual images, preserving their labels
    all_images = []
    all_labels = []

    for images, labels in dataset:
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())

    x_images = np.concatenate(all_images, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    print("Validation images:", len(y_true))

    print("\nActual images per class:")
    for class_id, class_name in enumerate(CLASS_NAMES):
        count = int(np.sum(y_true == class_id))
        print(f"{class_name}: {count}")

    model_files = [
        "tensorflow_document_model.keras",
        "tensorflow_document_model_improved.keras",
    ]

    # Compare each model on the same images
    for filename in model_files:
        print("\n" + "-" * 50)
        print("MODEL:", filename)
        print("-" * 50)

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            print("ERROR: Model file not found:", model_path)
            continue

        try:
            model = tf.keras.models.load_model(model_path)

            predictions = model.predict(
                x_images,
                batch_size=BATCH_SIZE,
                verbose=0,
            )

            if predictions.ndim != 2:
                print("ERROR: Unexpected prediction shape:", predictions.shape)
                continue

            if predictions.shape[1] != len(CLASS_NAMES):
                print(
                    "ERROR: This model predicts",
                    predictions.shape[1],
                    "classes, not 4."
                )
                continue

            y_pred = np.argmax(predictions, axis=1)

            correct = int(np.sum(y_true == y_pred))
            total = len(y_true)
            accuracy = correct / total if total else 0

            print(f"\nAccuracy: {accuracy * 100:.2f}%")
            print(f"Correct predictions: {correct}/{total}")

            print("\nPer-class accuracy:")
            for class_id, class_name in enumerate(CLASS_NAMES):
                mask = y_true == class_id
                class_total = int(np.sum(mask))

                if class_total == 0:
                    print(f"{class_name}: No validation images")
                    continue

                class_correct = int(
                    np.sum(y_pred[mask] == class_id)
                )

                print(
                    f"{class_name}: "
                    f"{class_correct}/{class_total} "
                    f"({class_correct / class_total * 100:.2f}%)"
                )

            print("\nPrediction counts:")
            for class_id, class_name in enumerate(CLASS_NAMES):
                predicted_count = int(np.sum(y_pred == class_id))
                print(f"{class_name}: {predicted_count}")

        except Exception as error:
            print("ERROR while evaluating this model:")
            print(error)

    print("\n" + "=" * 50)
    print("COMPARISON FINISHED")
    print("=" * 50)
    print(
        "Note: This is a preliminary result. "
        "The validation split must match the training setup "
        "before accuracy can be trusted."
    )


if __name__ == "__main__":
    main()