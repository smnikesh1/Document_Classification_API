import pytesseract
from PIL import Image
from pdf2image import convert_from_path


# -----------------------------------
# Tesseract OCR location
# -----------------------------------
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# -----------------------------------
# Poppler location
# -----------------------------------
POPPLER_PATH = (
    r"C:\Users\SAIRAM NIKESH\AppData\Local\Microsoft\WinGet\Packages"
    r"\oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\poppler-25.07.0\Library\bin"
)


# -----------------------------------
# OCR function
# -----------------------------------
def extract_text(file_path):

    # If the file is a PDF
    if file_path.lower().endswith(".pdf"):

        # Convert PDF pages into images
        pages = convert_from_path(
            file_path,
            poppler_path=POPPLER_PATH
        )

        text = ""

        # Run OCR on every page
        for page in pages:
            text += pytesseract.image_to_string(page)
            text += "\n"

        return text

    # If the file is JPG/PNG
    else:

        image = Image.open(file_path)

        text = pytesseract.image_to_string(image)

        return text