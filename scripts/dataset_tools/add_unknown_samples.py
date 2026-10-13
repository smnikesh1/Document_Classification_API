
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import textwrap

OUTPUT_DIR = Path("dataset/unknown")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

documents = [
    ("RESTAURANT RECEIPT", ["Restaurant: Sample Cafe", "Item: Veg Meals", "Quantity: 2", "Total: 400"]),
    ("LIBRARY CARD", ["Library: Sample Library", "Member: Test Student", "Category: Student", "Status: Active"]),
    ("BUS TICKET", ["Service: City Bus", "From: Central", "To: Airport", "Fare: 45"]),
    ("TRAIN BOOKING", ["Journey: Sample Route", "Class: Second Sitting", "Passengers: 1", "Status: Confirmed"]),
    ("DELIVERY RECEIPT", ["Order: TEST-1001", "Item: Books", "Delivery: Completed", "Amount: 650"]),
    ("PARKING RECEIPT", ["Location: Sample Parking", "Duration: 2 Hours", "Vehicle: TEST-1234", "Fee: 80"]),
    ("GYM MEMBERSHIP", ["Member: Test Person", "Plan: Monthly", "Start Year: 2026", "Status: Active"]),
    ("EVENT INVITATION", ["Event: Sample Technology Day", "Venue: City Hall", "Date: 15 November", "Entry: Free"]),
    ("MEETING NOTES", ["Topic: Weekly Meeting", "Team: Technology", "Action: Review tasks", "Status: Open"]),
    ("BOOK ORDER", ["Order: BOOK-1002", "Product: Python Guide", "Quantity: 1", "Amount: 550"]),
    ("FOOD ORDER", ["Order: FOOD-1003", "Item: Sandwich", "Quantity: 2", "Status: Delivered"]),
    ("MAINTENANCE REQUEST", ["Request: FIX-1004", "Location: Office Room 2", "Issue: Light repair", "Status: Pending"]),
    ("HOSTEL RECEIPT", ["Resident: Test Student", "Month: October", "Room: B-12", "Amount: 7000"]),
    ("COURSE TIMETABLE", ["Course: Computer Science", "Subject: Python", "Time: 10 AM", "Day: Monday"]),
    ("PROJECT SUMMARY", ["Project: Sample App", "Team: Development", "Phase: Testing", "Status: In Progress"]),
    ("RETURN RECEIPT", ["Order: RET-1005", "Product: Keyboard", "Reason: Damaged box", "Status: Received"]),
    ("WARRANTY CARD", ["Product: Sample Monitor", "Warranty: 1 Year", "Purchase Year: 2026", "Status: Active"]),
    ("VISITOR PASS", ["Visitor: Test Guest", "Department: Technology", "Visit: Meeting", "Status: Temporary"]),
    ("PACKAGE LABEL", ["Tracking: TEST-1006", "Contents: Stationery", "Destination: Office", "Status: Dispatched"]),
    ("CLASS ATTENDANCE", ["Class: Python Basics", "Student: Test Student", "Sessions: 20", "Attendance: 18"]),
    ("SPORTS REGISTRATION", ["Event: Sample Sports Day", "Participant: Test Player", "Category: Running", "Status: Registered"]),
    ("REPAIR ESTIMATE", ["Service: Laptop Repair", "Part: Keyboard", "Labour: 500", "Estimated Total: 2500"]),
    ("SHOPPING LIST", ["Item: Rice", "Item: Vegetables", "Item: Soap", "Purpose: Household"]),
    ("FEEDBACK FORM", ["Service: Sample Service", "Rating: 4 out of 5", "Comment: Good", "Status: Submitted"]),
    ("GENERAL NOTICE", ["Subject: Office Maintenance", "Date: 20 October", "Action: Use another entrance", "Issued By: Admin"]),
]

def load_font(size):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()

title_font = load_font(30)
body_font = load_font(23)
footer_font = load_font(16)

created = 0

for index, (title, lines) in enumerate(documents, start=6):
    output_path = OUTPUT_DIR / f"unknown_{index:02d}.png"

    # Do not overwrite any existing image.
    if output_path.exists():
        print(f"Skipped existing file: {output_path.name}")
        continue

    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)

    draw.rectangle((25, 25, 875, 575), outline="black", width=2)
    draw.text((55, 55), "SYNTHETIC SAMPLE DOCUMENT", font=footer_font, fill="gray")
    draw.text((55, 105), title, font=title_font, fill="black")
    draw.line((55, 155, 845, 155), fill="black", width=2)

    y = 195
    for line in lines:
        wrapped = textwrap.wrap(line, width=48)
        for part in wrapped:
            draw.text((65, y), part, font=body_font, fill="black")
            y += 42

    draw.line((55, 485, 845, 485), fill="gray", width=1)
    draw.text(
        (55, 510),
        "SYNTHETIC EXAMPLE - NOT A REAL IDENTITY OR GOVERNMENT DOCUMENT",
        font=footer_font,
        fill="gray",
    )

    image.save(output_path)
    created += 1
    print(f"Created: {output_path.name}")

print(f"\nFinished. New images created: {created}")
print(f"Unknown folder: {OUTPUT_DIR.resolve()}")