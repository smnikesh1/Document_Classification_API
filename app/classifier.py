def classify_document(text: str):

    text = text.lower()

    # ==========================================
    # Store evidence scores
    # ==========================================

    aadhar_score = 0.0
    pan_score = 0.0
    driving_score = 0.0
    passport_score = 0.0

    # ==========================================
    # Aadhaar evidence
    # ==========================================

    if "aadhaar" in text:
        aadhar_score += 0.50

    if "bharat" in text:
        aadhar_score += 0.15

    if "government" in text:
        aadhar_score += 0.10

    if "india" in text:
        aadhar_score += 0.10

    if "uidai" in text:
        aadhar_score += 0.15

    # ==========================================
    # PAN evidence
    # ==========================================

    if "permanent account number" in text:
        pan_score += 0.60

    if "income tax department" in text:
        pan_score += 0.25

    if "pan" in text:
        pan_score += 0.15

    # ==========================================
    # Driving Licence evidence
    # ==========================================

    if "driving licence" in text:
        driving_score += 0.70

    if "driving license" in text:
        driving_score += 0.70

    if "transport" in text:
        driving_score += 0.15

    if "licence" in text or "license" in text:
        driving_score += 0.15

    # ==========================================
    # Passport evidence
    # ==========================================

    if "passport" in text:
        passport_score += 0.70

    if "republic of india" in text:
        passport_score += 0.20

    if "nationality" in text:
        passport_score += 0.10

    # ==========================================
    # Compare all scores
    # ==========================================

    scores = {
        "aadhar": aadhar_score,
        "pan": pan_score,
        "driving_license": driving_score,
        "passport": passport_score
    }

    classified = max(scores, key=scores.get)
    confidence = scores[classified]

    # ==========================================
    # Unknown document
    # ==========================================

    if confidence < 0.40:
        return {
            "classified": "unknown",
            "confidence": round(confidence, 2)
        }

    # ==========================================
    # Final result
    # ==========================================

    return {
        "classified": classified,
        "confidence": round(confidence, 2)
    }