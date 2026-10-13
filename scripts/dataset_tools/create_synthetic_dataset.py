
from pathlib import Path
from random import Random

from PIL import Image, ImageDraw, ImageFont


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

TARGET_IMAGES_PER_CLASS = 30
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 620

random_generator = Random(42)

DOCUMENT_TYPES = {
    "aadhar": {
        "title": "SYNTHETIC AADHAAR-TYPE SAMPLE",
        "fields": [
            "Sample Identity Record",
            "Fictional Sample Name",
            "Sample Date of Birth",
            "Sample Address",
            "Training Record Number",
        ],
    },
    "pan": {
        "title": "SYNTHETIC PAN-TYPE SAMPLE",
        "fields": [
            "Sample Tax Record",
            "Fictional Sample Name",
            "Sample Date of Birth",
            "Sample Record Number",
            "Training Classification",
        ],
    },
    "driving_license": {
        "title": "SYNTHETIC DRIVING-LICENCE SAMPLE",
        "fields": [
            "Sample Driver Record",
            "Fictional Sample Name",
            "Sample Date of Birth",
            "Sample Vehicle Category",
            "Training Record Number",
        ],
    },
}

SAMPLE_NAMES = [
    "Aarav Sample",
    "Diya Example",
    "Kiran Demo",
    "Maya Training",
    "Arun Fictional",
    "Nila Example",
    "Ravi Sample",
    "Priya Demo",
    "Vijay Training",
    "Anu Fictional",
    "Dev Sample",
    "Meera Example",
    "Hari Demo",
    "Kavya Training",
    "Naveen Fictional",
]

SAMPLE_ADDRESSES = [
    "12 Example Street, Sample City",
    "25 Training Road, Demo Town",
    "8 Fictional Avenue, Example District",
    "44 Sample Lane, Training City",
    "17 Demo Street, Fictional Town",
]

SAMPLE_DATES = [
    "01-01-1995",
    "12-03-1998",
    "24-07-2000",
    "16-11-1996",
    "09-05-2001",
    "20-02-1994",
    "05-08-1999",
    "14-12-1997",
]

VEHICLE_CATEGORIES = [
    "Sample Category A",
    "Sample Category B",
    "Training Category C",
]


# ==================================================
# 2. LOAD FONTS
# ==================================================

def load_font(size):
    """Load a readable font, with a fallback if unavailable."""

    possible_fonts = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]

    for font_path in possible_fonts:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except OSError:
                pass

    return ImageFont.load_default()


# ==================================================
# 3. CREATE A SYNTHETIC DOCUMENT IMAGE
# ==================================================

def create_sample_image(document_type, sample_number):
    """Create a fictional, clearly watermarked training image."""

    info = DOCUMENT_TYPES[document_type]

    image = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        color=(250, 250, 245),
    )

    draw = ImageDraw.Draw(image)

    title_font = load_font(30)
    heading_font = load_font(23)
    body_font = load_font(21)
    watermark_font = load_font(28)
    footer_font = load_font(19)

    # Simple border, not an official identity-card design.
    draw.rectangle(
        (20, 20, IMAGE_WIDTH - 20, IMAGE_HEIGHT - 20),
        outline=(70, 90, 110),
        width=3,
    )

    # Header.
    draw.text(
        (45, 40),
        info["title"],
        fill=(25, 65, 100),
        font=title_font,
    )

    draw.line(
        (45, 95, IMAGE_WIDTH - 45, 95),
        fill=(120, 140, 155),
        width=2,
    )

    # Different content per image.
    name = SAMPLE_NAMES[
        (sample_number - 1) % len(SAMPLE_NAMES)
    ]

    address = SAMPLE_ADDRESSES[
        (sample_number - 1) % len(SAMPLE_ADDRESSES)
    ]

    date_of_birth = SAMPLE_DATES[
        (sample_number - 1) % len(SAMPLE_DATES)
    ]

    record_number = (
        f"SYNTHETIC-{document_type.upper()}-{sample_number:03d}"
    )

    values = [
        name,
        date_of_birth,
        address,
        record_number,
    ]

    if document_type == "driving_license":
        values.insert(
            3,
            VEHICLE_CATEGORIES[
                (sample_number - 1) % len(VEHICLE_CATEGORIES)
            ],
        )

    # Add a small amount of layout variation.
    left_margin = random_generator.choice([50, 60, 70])
    row_gap = random_generator.choice([52, 57, 62])
    top = random_generator.choice([125, 135, 145])

    for index, field_name in enumerate(info["fields"]):
        y = top + index * row_gap

        if index == 0:
            value = values[0]
        elif index == 1:
            value = values[1]
        elif index == 2:
            value = address
        elif index == 3:
            value = values[3] if len(values) > 3 else record_number
        else:
            value = record_number

        draw.text(
            (left_margin, y),
            f"{field_name}:",
            fill=(35, 50, 65),
            font=heading_font,
        )

        draw.text(
            (left_margin + 330, y),
            value,
            fill=(20, 20, 20),
            font=body_font,
        )

    # Prominent watermark: generated images are not real documents.
    watermark = "SYNTHETIC SAMPLE - NOT A REAL ID"

    draw.text(
        (55, 485),
        watermark,
        fill=(180, 40, 40),
        font=watermark_font,
    )

    draw.text(
        (55, 540),
        f"TRAINING IMAGE {sample_number:03d}",
        fill=(80, 80, 80),
        font=footer_font,
    )

    # Save as PNG.
    output_folder = DATASET_DIR / document_type
    output_folder.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_folder
        / f"{document_type}_synthetic_{sample_number:03d}.png"
    )

    image.save(output_path)

    return output_path


# ==================================================
# 4. ENSURE EACH FOLDER HAS 30 IMAGES
# ==================================================

def generate_dataset():
    print("\n========================================")
    print(" SYNTHETIC DOCUMENT DATASET GENERATOR")
    print("========================================\n")

    for document_type in DOCUMENT_TYPES:
        folder = DATASET_DIR / document_type
        folder.mkdir(parents=True, exist_ok=True)

        existing_images = [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff",
            }
        ]

        current_count = len(existing_images)

        print(f"\nDocument type: {document_type}")
        print(f"Existing images: {current_count}")

        if current_count >= TARGET_IMAGES_PER_CLASS:
            print(
                f"Already has {TARGET_IMAGES_PER_CLASS} or more "
                "images. No additional images created."
            )
            continue

        # The image numbering starts after the target's existing count.
        # Existing filenames are never overwritten.
        number = 1

        while current_count < TARGET_IMAGES_PER_CLASS:
            output_path = (
                folder
                / f"{document_type}_synthetic_{number:03d}.png"
            )

            if output_path.exists():
                number += 1
                continue

            create_sample_image(document_type, number)

            current_count += 1
            print(
                f"Created: {output_path.name} "
                f"({current_count}/{TARGET_IMAGES_PER_CLASS})"
            )

            number += 1

        print(f"Finished: {document_type} = {current_count} images")

    print("\n========================================")
    print(" DATASET GENERATION FINISHED")
    print("========================================")

    for document_type in DOCUMENT_TYPES:
        folder = DATASET_DIR / document_type

        image_count = sum(
            1
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff",
            }
        )

        print(f"{document_type}: {image_count} images")

    print(
        "\nIMPORTANT: These are synthetic training samples. "
        "They do not replace varied real-world evaluation data."
    )


# ==================================================
# 5. RUN PROGRAM
# ==================================================

if __name__ == "__main__":
    generate_dataset()