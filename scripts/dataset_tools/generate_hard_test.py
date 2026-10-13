from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


OUTPUT_DIR = Path("dataset/hard_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


try:
    font = ImageFont.truetype("arial.ttf", 28)
except:
    font = ImageFont.load_default()


# ============================================================
# TWO AMBIGUOUS DOCUMENTS
# These are ACTUALLY UNKNOWN documents.
# But they contain Aadhaar-related keywords.
# ============================================================

documents = [

    {
        "name": "hard_04.png",
        "text": """
        GENERAL INFORMATION DOCUMENT

        GOVERNMENT OF INDIA
        BHARAT
        INDIA

        This is a general information document.
        It is NOT an Aadhaar document.

        NOT A REAL GOVERNMENT DOCUMENT
        """
    },

    {
        "name": "hard_08.png",
        "text": """
        GENERAL GOVERNMENT NOTICE

        GOVERNMENT OF INDIA
        BHARAT
        INDIA

        This document contains general information.
        It is NOT an Aadhaar document.

        NOT A REAL GOVERNMENT DOCUMENT
        """
    }
]


# ============================================================
# CREATE THE TWO DOCUMENTS
# ============================================================

for document in documents:

    image = Image.new(
        "RGB",
        (1000, 700),
        "white"
    )

    draw = ImageDraw.Draw(image)

    y = 60

    for line in document["text"].strip().split("\n"):

        line = line.strip()

        if line:

            draw.text(
                (60, y),
                line,
                fill="black",
                font=font
            )

            y += 50

    output_path = OUTPUT_DIR / document["name"]

    image.save(output_path)

    print(f"Updated: {output_path}")


print("\nTwo ambiguous test documents created.")