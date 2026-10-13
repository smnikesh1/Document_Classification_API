
from pathlib import Path
import json
import tensorflow as tf

MODEL_PATH = Path("models/tensorflow_document_model.keras")
CLASS_NAMES_PATH = Path("models/tensorflow_class_names.json")
TEST_FOLDER = Path("dataset/hard_test")

# Expected labels inferred from the OCR text
expected_labels = {
    "hard_01.png": "aadhar",
    "hard_02.png": "pan",
    "hard_03.png": "driving_license",
    "hard_04.png": "unknown",
    "hard_05.png": "aadhar",
    "hard_06.png": "pan",
    "hard_07.png": "driving_license",
    "hard_08.png": "unknown",
    "hard_09.png": "aadhar",
    "hard_10.png": "pan",
    "hard_11.png": "driving_license",
    "hard_12.png": "unknown",
    "hard_13.png": "aadhar",
    "hard_14.png": "pan",
    "hard_15.png": "driving_license",
    "hard_16.png": "unknown",
    "hard_17.png": "aadhar",
    "hard_18.png": "pan",
    "hard_19.png": "driving_license",
    "hard_20.png": "unknown",
}

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

correct = 0
total = 0

print("\nHARD TEST EVALUATION")
print("-" * 65)

for filename, expected in expected_labels.items():
    image_path = TEST_FOLDER / filename

    if not image_path.is_file():
        print(f"{filename}: FILE NOT FOUND")
        continue

    image = tf.keras.utils.load_img(
        image_path, target_size=(224, 224)
    )
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, axis=0)

    scores = model.predict(image_array, verbose=0)[0]
    predicted = class_names[int(tf.argmax(scores))]
    confidence = float(scores.max()) * 100

    is_correct = predicted == expected
    correct += int(is_correct)
    total += 1

    status = "CORRECT" if is_correct else "WRONG"

    print(
        f"{filename}: expected={expected}, "
        f"predicted={predicted}, "
        f"confidence={confidence:.2f}%, {status}"
    )

print("-" * 65)

if total:
    print(f"Correct predictions: {correct}/{total}")
    print(f"Accuracy: {correct / total * 100:.2f}%")
else:
    print("No images were evaluated.")

print("\nNote: Expected labels are provisional and need verification.")