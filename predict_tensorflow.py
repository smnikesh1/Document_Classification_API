
from pathlib import Path
import json
import sys
import tensorflow as tf

MODEL_PATH = Path("models/tensorflow_document_model.keras")
CLASS_NAMES_PATH = Path("models/tensorflow_class_names.json")

if len(sys.argv) < 2:
    print("Usage: python predict_tensorflow.py path_to_image")
    raise SystemExit(1)

image_path = Path(sys.argv[1])

if not image_path.is_file():
    print("Error: Image file not found:", image_path)
    raise SystemExit(1)

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

image = tf.keras.utils.load_img(
    image_path,
    target_size=(224, 224),
)

image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, axis=0)

predictions = model.predict(image_array, verbose=0)[0]

best_index = int(tf.argmax(predictions))
predicted_class = class_names[best_index]
confidence = float(predictions[best_index]) * 100

print("\nPredicted document:", predicted_class)
print(f"Model confidence score: {confidence:.2f}%")

print("\nAll class scores:")
for name, score in zip(class_names, predictions):
    print(f"{name}: {score * 100:.2f}%")