
from pathlib import Path
import json
import tensorflow as tf

# ==================================================
# 1. SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
SEED = 42
EPOCHS = 20

CLASS_NAMES = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]

# Use NEW filenames to protect existing models.
MODEL_PATH = MODEL_DIR / "tensorflow_document_model_v4.keras"
CLASS_NAMES_PATH = MODEL_DIR / "tensorflow_class_names_v4.json"

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

MODEL_DIR.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)


# ==================================================
# 2. CHECK DATASET
# ==================================================

if not DATASET_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found: {DATASET_DIR}"
    )

for class_name in CLASS_NAMES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Missing category folder: {class_dir}"
        )

    image_files = [
        path for path in class_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    print(f"{class_name}: {len(image_files)} images")

    if len(image_files) < 5:
        raise ValueError(
            f"Add at least 5 images to {class_dir}"
        )

print("\nDataset folders checked successfully.")


# ==================================================
# 3. LOAD DATASET
# Only the four named folders are included.
# Other folders, such as hard_test, are excluded.
# ==================================================

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

print("\nClass order:", CLASS_NAMES)
print("Training batches:", len(train_data))
print("Validation batches:", len(validation_data))


# ==================================================
# 4. DATA AUGMENTATION
# Applied only while training.
# ==================================================

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomTranslation(0.03, 0.03),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="data_augmentation",
)


# ==================================================
# 5. LOAD PRETRAINED MOBILENETV2
# ==================================================

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

base_model.trainable = False


# ==================================================
# 6. BUILD MODEL
# ==================================================

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Keep pretrained model in inference mode.
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="document_category",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="document_classifier",
)


# ==================================================
# 7. COMPILE MODEL
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ==================================================
# 8. TRAINING CALLBACKS
# ==================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1,
    ),
]


# ==================================================
# 9. TRAIN MODEL
# ==================================================

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ==================================================
# 10. EVALUATE MODEL
# ==================================================

val_loss, val_accuracy = model.evaluate(
    validation_data,
    verbose=1,
)

print(f"\nValidation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.2%}")


# ==================================================
# 11. SAVE MODEL AND LABELS
# ==================================================

# Save the model first. Only write the labels if model
# saving succeeds.
model.save(MODEL_PATH)

with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\nTraining completed successfully.")
print(f"Model saved to: {MODEL_PATH}")
print(f"Class names saved to: {CLASS_NAMES_PATH}")
print("\nNo scikit-learn or joblib is used.")