
from pathlib import Path
import pandas as pd

from app.ocr import extract_text
from app.cleaner import clean_text
from app.classifier import classify_document_with_rules


# ============================================================
# 1. DATASET PATHS
# ============================================================

dataset_path = Path("dataset")

folders = {
    "aadhar": dataset_path / "aadhar",
    "pan": dataset_path / "pan",
    "driving_license": dataset_path / "driving_license",
    "unknown": dataset_path / "unknown",
    "hard_test": dataset_path / "hard_test",
}


# ============================================================
# 2. STORE ALL RESULTS
# ============================================================

results = []


# ============================================================
# 3. HELPER FUNCTION TO PROCESS A DOCUMENT
# ============================================================

def process_document(file_path, actual_label, dataset_name):
    """
    Extract text, clean it, classify the document,
    and store the prediction result.
    """

    try:
        # Step 1: Extract text using OCR
        extracted_text = extract_text(str(file_path))

        # Step 2: Clean the extracted text
        cleaned_text = clean_text(extracted_text)

        # Step 3: Classify using rules (no Gemini API call)
        result = classify_document_with_rules(cleaned_text)

        predicted = result["classified"]
        confidence = result["confidence"]

        # Step 4: Compare actual and predicted labels
        prediction_result = (
            "Correct" if actual_label == predicted else "Wrong"
        )

        results.append({
            "File": file_path.name,
            "Dataset": dataset_name,
            "Actual": actual_label,
            "Predicted": predicted,
            "Confidence": confidence,
            "Result": prediction_result,
        })

        print(
            f"{file_path.name:<25} "
            f"Actual: {actual_label:<18} "
            f"Predicted: {predicted:<18} "
            f"{prediction_result}"
        )

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")

        results.append({
            "File": file_path.name,
            "Dataset": dataset_name,
            "Actual": actual_label,
            "Predicted": "ERROR",
            "Confidence": 0,
            "Result": "Error",
        })


# 4. TEST NORMAL DATASETS

print("\nProcessing normal datasets...")
print("-" * 90)

normal_labels = [
    "aadhar",
    "pan",
    "driving_license",
    "unknown",
]

for actual_label in normal_labels:
    folder = folders[actual_label]

    if not folder.exists():
        print(f"Warning: Dataset folder not found: {folder}")
        continue

    for file_path in sorted(folder.iterdir()):
        if file_path.is_file():
            process_document(
                file_path=file_path,
                actual_label=actual_label,
                dataset_name=actual_label,
            )


 
# 5. HARD TEST DATASET LABELS


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
    "hard_20.png": "unknown",
}



# 6. TEST HARD DATASET

print("\nProcessing hard-test dataset...")
print("-" * 90)

hard_test_folder = folders["hard_test"]

if not hard_test_folder.exists():
    print(f"Warning: Hard-test folder not found: {hard_test_folder}")

for file_name, actual_label in hard_test_labels.items():
    file_path = hard_test_folder / file_name

    if not file_path.is_file():
        print(f"Warning: Hard-test file not found: {file_path}")

        results.append({
            "File": file_name,
            "Dataset": "hard_test",
            "Actual": actual_label,
            "Predicted": "ERROR",
            "Confidence": 0,
            "Result": "Error",
        })

        continue

    process_document(
        file_path=file_path,
        actual_label=actual_label,
        dataset_name="hard_test",
    )


# ============================================================
# 7. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    results,
    columns=[
        "File",
        "Dataset",
        "Actual",
        "Predicted",
        "Confidence",
        "Result",
    ],
)


# ============================================================
# 8. CALCULATE BASIC METRICS
# ============================================================

total = len(df)

correct = int(
    ((df["Actual"] == df["Predicted"]) &
     (df["Predicted"] != "ERROR")).sum()
)

errors = int((df["Predicted"] == "ERROR").sum())

wrong = int(
    ((df["Actual"] != df["Predicted"]) &
     (df["Predicted"] != "ERROR")).sum()
)

# Processing errors are counted in total but not as correct predictions.
accuracy = correct / total if total > 0 else 0


# ============================================================
# 9. CALCULATE WEIGHTED PRECISION, RECALL AND F1
# ============================================================

# Use only successfully classified documents for class metrics.
valid_df = df[df["Predicted"] != "ERROR"].copy()

labels = sorted(valid_df["Actual"].unique())

valid_total = len(valid_df)

weighted_precision = 0.0
weighted_recall = 0.0
weighted_f1 = 0.0

for label in labels:

    # Number of actual documents belonging to this class
    actual_count = int((valid_df["Actual"] == label).sum())

    # True positives
    true_positive = int(
        ((valid_df["Actual"] == label) &
         (valid_df["Predicted"] == label)).sum()
    )

    # Number of documents predicted as this class
    predicted_count = int(
        (valid_df["Predicted"] == label).sum()
    )

    # Precision
    class_precision = (
        true_positive / predicted_count
        if predicted_count > 0
        else 0.0
    )

    # Recall
    class_recall = (
        true_positive / actual_count
        if actual_count > 0
        else 0.0
    )

    # F1 score
    denominator = class_precision + class_recall

    class_f1 = (
        2 * class_precision * class_recall / denominator
        if denominator > 0
        else 0.0
    )

    # Weighted values
    weighted_precision += class_precision * actual_count
    weighted_recall += class_recall * actual_count
    weighted_f1 += class_f1 * actual_count


precision = (
    weighted_precision / valid_total
    if valid_total > 0
    else 0.0
)

recall = (
    weighted_recall / valid_total
    if valid_total > 0
    else 0.0
)

f1 = (
    weighted_f1 / valid_total
    if valid_total > 0
    else 0.0
)


# ============================================================
# 10. CREATE SUMMARY SHEET
# ============================================================

summary = pd.DataFrame({
    "Metric": [
        "Total Documents",
        "Correct Predictions",
        "Wrong Predictions",
        "Processing Errors",
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1 Score",
    ],
    "Value": [
        total,
        correct,
        wrong,
        errors,
        f"{accuracy * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{recall * 100:.2f}%",
        f"{f1 * 100:.2f}%",
    ],
})


# ============================================================
# 11. CREATE WRONG PREDICTIONS SHEET
# ============================================================

wrong_predictions = df[
    (df["Actual"] != df["Predicted"]) &
    (df["Predicted"] != "ERROR")
].copy()


# ============================================================
# 12. CREATE EXCEL REPORT
# ============================================================

output_file = Path("classification_report.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False,
    )

    df.to_excel(
        writer,
        sheet_name="Prediction Details",
        index=False,
    )

    wrong_predictions.to_excel(
        writer,
        sheet_name="Wrong Predictions",
        index=False,
    )


# ============================================================
# 13. PRINT FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("DOCUMENT CLASSIFICATION REPORT")
print("=" * 60)

print(f"\nTotal Documents : {total}")
print(f"Correct         : {correct}")
print(f"Wrong           : {wrong}")
print(f"Errors          : {errors}")

print(f"\nAccuracy        : {accuracy * 100:.2f}%")
print(f"Precision       : {precision * 100:.2f}%")
print(f"Recall          : {recall * 100:.2f}%")
print(f"F1 Score        : {f1 * 100:.2f}%")

print("\n" + "=" * 60)
print(f"Excel report created: {output_file.resolve()}")
print("=" * 60)
