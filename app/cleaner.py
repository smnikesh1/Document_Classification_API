import re


def clean_text(text: str):
    # Convert everything to lowercase
    text = text.lower()

    # Replace new lines with spaces
    text = text.replace("\n", " ")

    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove spaces from beginning and end
    text = text.strip()

    return text