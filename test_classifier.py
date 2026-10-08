from app.ocr import extract_text
from app.cleaner import clean_text
from app.classifier import classify_document

file_path = r"C:\Users\SAIRAM NIKESH\Downloads\GovermentId.pdf"

print("STARTING...")

text = extract_text(file_path)

print("===== RAW OCR TEXT =====")
print(text)

cleaned_text = clean_text(text)

print("===== CLEANED TEXT =====")
print(cleaned_text)

result = classify_document(cleaned_text)

print("===== CLASSIFICATION RESULT =====")
print(result)

print("DONE")
