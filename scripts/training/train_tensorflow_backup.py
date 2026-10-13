
from pathlib import Path
import json
import tensorflow as tf

# Settings
DATASET_DIR = Path("dataset")
MODEL_DIR = Path("models")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42
EPOCHS = 10

CLASS_NAMES = [
    "aadhar",
    "driving_license",
    "pan",
    "unknown",
]

MODEL_DIR.mkdir(exist_ok=True)

# Load training images
train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Load validation images
validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    class_names=CLASS_NAMES,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

# Improve image-loading performance
train_data = train_data.prefetch(tf.data.AUTOTUNE)
validation_data = validation_data.prefetch(tf.data.AUTOTUNE)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

# Build our document classifier
inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

# Configure learning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Train
model.summary()

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
)

# Save TensorFlow model separately from XGBoost
model.save(MODEL_DIR / "tensorflow_document_model.keras")

with open(MODEL_DIR / "tensorflow_class_names.json", "w") as file:
    json.dump(CLASS_NAMES, file, indent=4)

print("\nTraining completed!")
print("Model saved to models/tensorflow_document_model.keras")
print("Class names saved to models/tensorflow_class_names.json")
print("Existing XGBoost model was not modified.")