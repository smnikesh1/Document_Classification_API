
from pathlib import Path
import json
import tensorflow as tf

# Project folders
BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Training settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 4
SEED = 42
EPOCHS = 30

CLASS_NAMES = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]

print("TensorFlow version:", tf.__version__)
print("Dataset folder:", DATASET_DIR)

if not DATASET_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

# Load training and validation images
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
)

print("Detected classes:", train_ds.class_names)

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Mild image augmentation
augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.03),
        tf.keras.layers.RandomZoom(0.08),
        tf.keras.layers.RandomTranslation(0.04, 0.04),
    ],
    name="data_augmentation",
)

# Pretrained MobileNetV2 feature extractor
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Keep pretrained layers frozen for this small dataset
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# Stop when validation loss stops improving
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

# Save as v5: do not overwrite v4
model_path = MODEL_DIR / "tensorflow_document_model_v5.keras"
classes_path = MODEL_DIR / "tensorflow_class_names_v5.json"

model.save(model_path)

with open(classes_path, "w", encoding="utf-8") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\nTraining completed.")
print("Model saved to:", model_path)
print("Class names saved to:", classes_path)
print("Best validation accuracy:", max(history.history["val_accuracy"]))