from pathlib import Path
import pandas as pd

from app.ocr import extract_text
from app.cleaner import clean_text
from app.classifier import classify_document


# ============================================================
# 1. DATASET LOCATION
# ============================================================

DATASET_DIR = Path("dataset")


# ============================================================
# 2. NORMAL DATASET FOLDERS
# ============================================================

folders = {
    "aadhar": DATASET_DIR / "aadhar",
    "pan": DATASET_DIR / "pan",
    "driving_license": DATASET_DIR / "driving_license",
    "unknown": DATASET_DIR / "unknown"
}


# ============================================================
# 3. STORE ALL RESULTS
# ============================================================

results = []


# ============================================================
# 4. PROCESS NORMAL DATASET
# ============================================================

for actual_class, folder in folders.items():

    print(f"\nProcessing {actual_class.upper()} documents...")

    for file_path in folder.glob("*.png"):

        print(f"Processing: {file_path.name}")

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        raw_text = extract_text(str(file_path))

        # ----------------------------------------------------
        # Clean OCR text
        # ----------------------------------------------------

        cleaned_text = clean_text(raw_text)

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        prediction = classify_document(cleaned_text)

        predicted_class = prediction["classified"]
        confidence = prediction["confidence"]

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({
            "File Name": file_path.name,
            "Actual Value": actual_class,
            "Predicted Value": predicted_class,
            "Confidence": confidence,
            "OCR Text": raw_text
        })


# ============================================================
# 5. HARD TEST DATASET
# ============================================================

hard_test_folder = DATASET_DIR / "hard_test"


# ============================================================
# 6. EXPECTED LABEL FOR EACH HARD TEST FILE
# ============================================================

hard_test_labels = {

    "hard_01.png": "aadhar",
    "hard_02.png": "pan",
    "hard_03.png": "driving_license",
    "hard_04.png": "unknown",

    "hard_05.png": "aadhar",
    "hard_06.png": "pan",
    "hard_07.png": "driving_license",
    "hard_08.png": "unknown",

    "hard_09.png": "aadhar",
    "hard_10.png": "pan",
    "hard_11.png": "driving_license",
    "hard_12.png": "unknown",

    "hard_13.png": "aadhar",
    "hard_14.png": "pan",
    "hard_15.png": "driving_license",
    "hard_16.png": "unknown",

    "hard_17.png": "aadhar",
    "hard_18.png": "pan",
    "hard_19.png": "driving_license",
    "hard_20.png": "unknown"
}


# ============================================================
# 7. PROCESS HARD TEST DOCUMENTS
# ============================================================

print("\nProcessing HARD TEST documents...")


for file_path in hard_test_folder.glob("*.png"):

    print(f"Processing: {file_path.name}")

    # --------------------------------------------------------
    # OCR
    # --------------------------------------------------------

    raw_text = extract_text(str(file_path))

    # --------------------------------------------------------
    # Clean OCR text
    # --------------------------------------------------------

    cleaned_text = clean_text(raw_text)

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    prediction = classify_document(cleaned_text)

    predicted_class = prediction["classified"]
    confidence = prediction["confidence"]

    # --------------------------------------------------------
    # Get actual class
    # --------------------------------------------------------

    actual_class = hard_test_labels[file_path.name]

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    results.append({
        "File Name": file_path.name,
        "Actual Value": actual_class,
        "Predicted Value": predicted_class,
        "Confidence": confidence,
        "OCR Text": raw_text
    })


# ============================================================
# 8. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 9. CHECK TOTAL DOCUMENTS
# ============================================================

total = len(df)


# ============================================================
# 10. CALCULATE CORRECT PREDICTIONS
# ============================================================

correct = sum(
    df["Actual Value"] == df["Predicted Value"]
)


# ============================================================
# 11. CALCULATE WRONG PREDICTIONS
# ============================================================

wrong = total - correct


# ============================================================
# 12. CALCULATE ACCURACY
# ============================================================

if total > 0:

    accuracy = correct / total

else:

    accuracy = 0


# ============================================================
# 13. CLASSES
# ============================================================

classes = [
    "aadhar",
    "pan",
    "driving_license",
    "unknown"
]


# ============================================================
# 14. STORE METRIC VALUES
# ============================================================

precision_values = []
recall_values = []
f1_values = []
support_values = []


# ============================================================
# 15. CALCULATE PRECISION / RECALL / F1
# ============================================================

for class_name in classes:

    # --------------------------------------------------------
    # Actual documents belonging to this class
    # --------------------------------------------------------

    actual_positive = sum(
        df["Actual Value"] == class_name
    )


    # --------------------------------------------------------
    # True Positive
    # --------------------------------------------------------

    true_positive = sum(
        (df["Actual Value"] == class_name)
        &
        (df["Predicted Value"] == class_name)
    )


    # --------------------------------------------------------
    # False Positive
    # --------------------------------------------------------

    false_positive = sum(
        (df["Actual Value"] != class_name)
        &
        (df["Predicted Value"] == class_name)
    )


    # --------------------------------------------------------
    # False Negative
    # --------------------------------------------------------

    false_negative = sum(
        (df["Actual Value"] == class_name)
        &
        (df["Predicted Value"] != class_name)
    )


    # ========================================================
    # PRECISION
    # ========================================================

    if true_positive + false_positive > 0:

        precision = true_positive / (
            true_positive + false_positive
        )

    else:

        precision = 0


    # ========================================================
    # RECALL
    # ========================================================

    if true_positive + false_negative > 0:

        recall = true_positive / (
            true_positive + false_negative
        )

    else:

        recall = 0


    # ========================================================
    # F1 SCORE
    # ========================================================

    if precision + recall > 0:

        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    else:

        f1 = 0


    # --------------------------------------------------------
    # Store values
    # --------------------------------------------------------

    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)
    support_values.append(actual_positive)


# ============================================================
# 16. WEIGHTED METRICS
# ============================================================

total_support = sum(support_values)


if total_support > 0:

    weighted_precision = sum(
        precision * support
        for precision, support
        in zip(
            precision_values,
            support_values
        )
    ) / total_support


    weighted_recall = sum(
        recall * support
        for recall, support
        in zip(
            recall_values,
            support_values
        )
    ) / total_support


    weighted_f1 = sum(
        f1 * support
        for f1, support
        in zip(
            f1_values,
            support_values
        )
    ) / total_support

else:

    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0


# ============================================================
# 17. ADD RESULT STATUS
# ============================================================

df["Status"] = df.apply(
    lambda row:
    "Correct"
    if row["Actual Value"] == row["Predicted Value"]
    else "Wrong",
    axis=1
)


# ============================================================
# 18. CREATE SUMMARY TABLE
# ============================================================

summary = pd.DataFrame({

    "Metric": [
        "Total Documents",
        "Correct Predictions",
        "Wrong Predictions",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Value": [
        total,
        correct,
        wrong,
        accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1
    ]
})


# ============================================================
# 19. CREATE EXCEL REPORT
# ============================================================

output_file = "classification_report.xlsx"


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    # --------------------------------------------------------
    # Document Results Sheet
    # --------------------------------------------------------

    df.to_excel(
        writer,
        sheet_name="Document Results",
        index=False
    )

    # --------------------------------------------------------
    # Metrics Sheet
    # --------------------------------------------------------

    summary.to_excel(
        writer,
        sheet_name="Metrics",
        index=False
    )


# ============================================================
# 20. PRINT CLASSIFICATION REPORT
# ============================================================

print("\n")
print("==============================================")
print("          CLASSIFICATION REPORT")
print("==============================================")


print(
    df[
        [
            "File Name",
            "Actual Value",
            "Predicted Value",
            "Confidence",
            "Status"
        ]
    ].to_string(index=False)
)


# ============================================================
# 21. PRINT WRONG PREDICTIONS ONLY
# ============================================================

wrong_documents = df[
    df["Actual Value"] != df["Predicted Value"]
]


print("\n")
print("==============================================")
print("          WRONG PREDICTIONS")
print("==============================================")


if len(wrong_documents) > 0:

    print(
        wrong_documents[
            [
                "File Name",
                "Actual Value",
                "Predicted Value",
                "Confidence"
            ]
        ].to_string(index=False)
    )

else:

    print("No wrong predictions.")


# ============================================================
# 22. PRINT METRICS
# ============================================================

print("\n")
print("==============================================")
print("                 METRICS")
print("==============================================")


print(f"Total Documents : {total}")
print(f"Correct         : {correct}")
print(f"Wrong           : {wrong}")

print(f"Accuracy        : {accuracy:.2%}")
print(f"Precision       : {weighted_precision:.2%}")
print(f"Recall          : {weighted_recall:.2%}")
print(f"F1 Score        : {weighted_f1:.2%}")


# ============================================================
# 23. EXCEL CONFIRMATION
# ============================================================

print("\n")
print("==============================================")

print(
    f"Excel report created: {output_file}"
)

print("==============================================")