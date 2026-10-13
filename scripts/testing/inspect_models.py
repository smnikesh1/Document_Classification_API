
from pathlib import Path
import tensorflow as tf

MODEL_DIR = Path("models")

model_files = [
    "tensorflow_document_model.keras",
    "tensorflow_document_model_improved.keras",
]

for filename in model_files:
    print("\n" + "=" * 60)
    print("MODEL:", filename)
    print("=" * 60)

    path = MODEL_DIR / filename

    if not path.exists():
        print("Model file not found:", path)
        continue

    try:
        model = tf.keras.models.load_model(path)

        print("Input shape:", model.input_shape)
        print("Output shape:", model.output_shape)
        print("Total layers:", len(model.layers))

        print("\nLast 8 layers:")
        for layer in model.layers[-8:]:
            print(
                layer.name,
                "|",
                layer.__class__.__name__,
                "| output:",
                getattr(layer, "output", None).shape
                if hasattr(getattr(layer, "output", None), "shape")
                else "not available",
            )

    except Exception as error:
        print("Could not inspect model:", error)