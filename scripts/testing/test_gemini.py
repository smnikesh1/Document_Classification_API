
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("ERROR: GEMINI_API_KEY was not loaded.")
else:
    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
           model="gemini-3.8-flash",
            contents="Reply with exactly: Connection successful",
        )

        print("SUCCESS:", response.text)

    except Exception as error:
        message = str(error)

        # Never display the API key.
        message = message.replace(api_key, "[REDACTED]")

        print("ERROR TYPE:", type(error).__name__)
        print("ERROR DETAILS:", message)