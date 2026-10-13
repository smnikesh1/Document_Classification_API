from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


# Output folder

output_folder = Path("dataset/unknown")
output_folder.mkdir(parents=True, exist_ok=True)


# Image settings

width = 1000
height = 600


# Font

font_path = "C:/Windows/Fonts/arial.ttf"

title_font = ImageFont.truetype(
    font_path,
    42
)

heading_font = ImageFont.truetype(
    font_path,
    32
)

normal_font = ImageFont.truetype(
    font_path,
    26
)


# Different unknown document types

documents = [
    {
        "title": "SHOPPING INVOICE",
        "heading": "PURCHASE INVOICE",
        "details": [
            "Customer: TEST CUSTOMER",
            "Product: Laptop",
            "Quantity: 1",
            "Total Amount: 45000"
        ]
    },

    {
        "title": "COLLEGE CERTIFICATE",
        "heading": "COURSE COMPLETION CERTIFICATE",
        "details": [
            "Student: TEST STUDENT",
            "Course: Computer Science",
            "Year: 2025",
            "Status: Completed"
        ]
    },

    {
        "title": "UTILITY BILL",
        "heading": "ELECTRICITY BILL",
        "details": [
            "Customer: TEST CUSTOMER",
            "Connection: TEST12345",
            "Billing Month: September",
            "Amount: 1250"
        ]
    },

    {
        "title": "BANK STATEMENT",
        "heading": "ACCOUNT STATEMENT",
        "details": [
            "Account Holder: TEST CUSTOMER",
            "Account Type: Savings",
            "Statement Period: 2026",
            "Balance: 25000"
        ]
    },

    {
        "title": "OFFICE DOCUMENT",
        "heading": "EMPLOYEE INFORMATION",
        "details": [
            "Employee: TEST EMPLOYEE",
            "Department: Technology",
            "Designation: Developer",
            "Status: Active"
        ]
    }
]


# Create documents

for i, document in enumerate(documents, start=1): #Give item and number for each item 

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

    # Synthetic label(Computer Code Generated)

    draw.text(
        (250, 50),
        "SYNTHETIC TEST DOCUMENT",
        fill="black",
        font=heading_font
    )

    # Document title

    draw.text(
        (300, 110),
        document["title"],
        fill="black",
        font=title_font
    )

    # Heading
    

    draw.text(
        (80, 190),
        document["heading"],
        fill="black",
        font=heading_font
    )

    # Details
    

    y_position = 260

    for detail in document["details"]:

        draw.text(
            (80, y_position),
            detail,
            fill="black",
            font=normal_font
        )

        y_position += 50

   #safety label

    draw.text(
        (80, 500),
        "NOT A REAL GOVERNMENT DOCUMENT",
        fill="black",
        font=normal_font
    )

   
    #save
    output_file = (
        output_folder /
        f"unknown_{i:02d}.png"
    )

    image.save(output_file)

    print(f"Created: {output_file}")


print("\n5 synthetic unknown documents created.")