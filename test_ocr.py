print("TEST STARTED")

from app.ocr import extract_text

file_path = r"C:\Users\SAIRAM NIKESH\Downloads\GovermentId.pdf"

print("FILE PATH:", file_path)

text = extract_text(file_path)

print("===== OCR RESULT =====")
print(text)

print("TEST FINISHED")
