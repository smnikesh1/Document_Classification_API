from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


# Output folder
OUTPUT_DIR = Path("dataset/pan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Synthetic test data
documents = [
    {
        "name": "Test Person 01",
        "dob": "01/01/2000"
    },
    {
        "name": "Sample Person 02",
        "dob": "15/03/1998"
    },
    {
        "name": "Demo Person 03",
        "dob": "22/07/1995"
    },
    {
        "name": "Synthetic Person 04",
        "dob": "10/11/2001"
    },
    {
        "name": "Test User 05",
        "dob": "05/06/1999"
    }
]


# Font
font_path = r"C:\Windows\Fonts\arial.ttf"

title_font = ImageFont.truetype(font_path, 36)
heading_font = ImageFont.truetype(font_path, 28)
normal_font = ImageFont.truetype(font_path, 24)


# Generate 5 synthetic PAN images
for index, person in enumerate(documents, start=1):

    image = Image.new(
        "RGB",
        (1000, 650),
        "white"
    )

    draw = ImageDraw.Draw(image)

    draw.text(
        (50, 40),
        "SYNTHETIC DOCUMENT - TEST DATA",
        font=title_font,
        fill="black"
    )

    # Important PAN classification words
    draw.text(
        (50, 120),
        "INCOME TAX DEPARTMENT",
        font=heading_font,
        fill="black"
    )

    draw.text(
        (50, 170),
        "GOVERNMENT OF INDIA",
        font=heading_font,
        fill="black"
    )

    draw.text(
        (50, 230),
        "PERMANENT ACCOUNT NUMBER CARD",
        font=heading_font,
        fill="black"
    )

    # Fake PAN value
    draw.text(
        (50, 310),
        "PAN: TESTP0000X",
        font=normal_font,
        fill="black"
    )

    draw.text(
        (50, 370),
        f"Name: {person['name']}",
        font=normal_font,
        fill="black"
    )

    draw.text(
        (50, 420),
        f"Date of Birth: {person['dob']}",
        font=normal_font,
        fill="black"
    )

    draw.text(
        (50, 550),
        "NOT A REAL GOVERNMENT DOCUMENT",
        font=normal_font,
        fill="black"
    )

    output_file = OUTPUT_DIR / f"pan_{index:02d}.png"

    image.save(output_file)

    print(f"Created: {output_file}")


print("\n5 synthetic PAN test images created successfully!")