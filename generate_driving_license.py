from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


# Output folder

output_folder = Path("dataset/driving_license")
output_folder.mkdir(parents=True, exist_ok=True)


# Image settings

width = 1000
height = 600


# Font

font_path = "C:/Windows/Fonts/arial.ttf"

title_font = ImageFont.truetype(font_path, 42)
heading_font = ImageFont.truetype(font_path, 32)
normal_font = ImageFont.truetype(font_path, 26)


# Create 5 synthetic documents

for i in range(1, 6):

    image = Image.new(
        "RGB",
        (width, height),
        "white"
    )

    draw = ImageDraw.Draw(image)

    # Border
    draw.rectangle(
        (20, 20, width - 20, height - 20),
        outline="black",
        width=4
    )

    # Title
    draw.text(
        (250, 50),
        "SYNTHETIC TEST DOCUMENT",
        fill="black",
        font=heading_font
    )

    draw.text(
        (330, 110),
        "DRIVING LICENCE",
        fill="black",
        font=title_font
    )

    # Information
    draw.text(
        (80, 200),
        "DRIVING LICENCE",
        fill="black",
        font=heading_font
    )

    draw.text(
        (80, 260),
        "Name: TEST PERSON",
        fill="black",
        font=normal_font
    )

    draw.text(
        (80, 310),
        "Licence Number: TESTDL0000",
        fill="black",
        font=normal_font
    )

    draw.text(
        (80, 360),
        "Date of Birth: 01-01-1995",
        fill="black",
        font=normal_font
    )

    draw.text(
        (80, 410),
        "Transport Department",
        fill="black",
        font=normal_font
    )

    # Safety label
    draw.text(
        (80, 500),
        "NOT A REAL GOVERNMENT DOCUMENT",
        fill="black",
        font=normal_font
    )

    # Save
    output_file = output_folder / f"driving_license_{i:02d}.png"

    image.save(output_file)

    print(f"Created: {output_file}")


print("\n5 synthetic driving licence documents created.")