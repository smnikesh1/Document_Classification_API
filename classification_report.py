from pathlib import Path
import pandas as pd

from app.ocr import extract_text
from app.cleaner import clean_text
from app.classifier import classify_document


# ============================================================
# 1. DATASET PATHS
# ============================================================

dataset_path = Path("dataset")

folders = {
    "aadhar": dataset_path / "aadhar",
    "pan": dataset_path / "pan",
    "driving_license": dataset_path / "driving_license",
    "unknown": dataset_path / "unknown",
    "hard_test": dataset_path / "hard_test"
}


# ============================================================
# 2. STORE ALL RESULTS
# ============================================================

results = []


# ============================================================
# 3. TEST NORMAL DATASETS
# ============================================================

for actual_label in ["aadhar", "pan", "driving_license", "unknown"]:

    folder = folders[actual_label]

    for file_path in folder.glob("*"):

        try:

            # OCR
            extracted_text = extract_text(str(file_path))

            # Clean OCR text
            cleaned_text = clean_text(extracted_text)

            # Classification
            result = classify_document(cleaned_text)

            predicted = result["classified"]
            confidence = result["confidence"]

            # Check prediction
            if actual_label == predicted:
                prediction_result = "Correct"
            else:
                prediction_result = "Wrong"

            results.append({
                "File": file_path.name,
                "Dataset": actual_label,
                "Actual": actual_label,
                "Predicted": predicted,
                "Confidence": confidence,
                "Result": prediction_result
            })

        except Exception as e:

            print(f"Error processing {file_path.name}: {e}")

            results.append({
                "File": file_path.name,
                "Dataset": actual_label,
                "Actual": actual_label,
                "Predicted": "ERROR",
                "Confidence": 0,
                "Result": "Error"
            })


# ============================================================
# 4. HARD TEST DATASET LABELS
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
# 5. TEST HARD DATASET
# ============================================================

hard_test_folder = folders["hard_test"]


for file_name, actual_label in hard_test_labels.items():

    file_path = hard_test_folder / file_name

    try:

        # OCR
        extracted_text = extract_text(str(file_path))

        # Clean text
        cleaned_text = clean_text(extracted_text)

        # Classification
        result = classify_document(cleaned_text)

        predicted = result["classified"]
        confidence = result["confidence"]

        # Check prediction
        if actual_label == predicted:
            prediction_result = "Correct"
        else:
            prediction_result = "Wrong"

        results.append({
            "File": file_name,
            "Dataset": "hard_test",
            "Actual": actual_label,
            "Predicted": predicted,
            "Confidence": confidence,
            "Result": prediction_result
        })

    except Exception as e:

        print(f"Error processing {file_name}: {e}")

        results.append({
            "File": file_name,
            "Dataset": "hard_test",
            "Actual": actual_label,
            "Predicted": "ERROR",
            "Confidence": 0,
            "Result": "Error"
        })


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(results)


# ============================================================
# 7. CALCULATE BASIC METRICS
# ============================================================

total = len(df)

correct = (df["Actual"] == df["Predicted"]).sum()

wrong = (
    (df["Actual"] != df["Predicted"]) &
    (df["Predicted"] != "ERROR")
).sum()

errors = (df["Predicted"] == "ERROR").sum()


# Accuracy

accuracy = correct / total if total > 0 else 0


# ============================================================
# 8. CALCULATE WEIGHTED PRECISION / RECALL / F1
# ============================================================

labels = sorted(
    set(df["Actual"])
)


weighted_precision = 0
weighted_recall = 0
weighted_f1 = 0


for label in labels:

    # Number of actual documents belonging to this class
    actual_count = (
        df["Actual"] == label
    ).sum()


    # True positives
    true_positive = (
        (df["Actual"] == label) &
        (df["Predicted"] == label)
    ).sum()


    # Number predicted as this class
    predicted_count = (
        df["Predicted"] == label
    ).sum()


    # Precision
    if predicted_count > 0:

        class_precision = (
            true_positive /
            predicted_count
        )

    else:

        class_precision = 0


    # Recall
    if actual_count > 0:

        class_recall = (
            true_positive /
            actual_count
        )

    else:

        class_recall = 0


    # F1 Score
    if (
        class_precision +
        class_recall
    ) > 0:

        class_f1 = (
            2 *
            class_precision *
            class_recall
        ) / (
            class_precision +
            class_recall
        )

    else:

        class_f1 = 0


    # Weighted values
    weighted_precision += (
        class_precision *
        actual_count
    )

    weighted_recall += (
        class_recall *
        actual_count
    )

    weighted_f1 += (
        class_f1 *
        actual_count
    )


# Final weighted metrics

precision = (
    weighted_precision / total
    if total > 0 else 0
)

recall = (
    weighted_recall / total
    if total > 0 else 0
)

f1 = (
    weighted_f1 / total
    if total > 0 else 0
)


# ============================================================
# 9. CREATE SUMMARY SHEET
# ============================================================

summary = pd.DataFrame({

    "Metric": [

        "Total Documents",

        "Correct Predictions",

        "Wrong Predictions",

        "Processing Errors",

        "Accuracy",

        "Precision",

        "Recall",

        "F1 Score"

    ],

    "Value": [

        total,

        correct,

        wrong,

        errors,

        f"{accuracy * 100:.2f}%",

        f"{precision * 100:.2f}%",

        f"{recall * 100:.2f}%",

        f"{f1 * 100:.2f}%"

    ]

})


# ============================================================
# 10. CREATE WRONG PREDICTIONS SHEET
# ============================================================

wrong_predictions = df[
    (df["Actual"] != df["Predicted"]) &
    (df["Predicted"] != "ERROR")
].copy()


# ============================================================
# 11. CREATE EXCEL REPORT
# ============================================================

output_file = "classification_report.xlsx"


with pd.ExcelWriter(
    output_file,
    engine="openpyxl"
) as writer:

    # Summary
    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    # All predictions
    df.to_excel(
        writer,
        sheet_name="Prediction Details",
        index=False
    )

    # Wrong predictions
    wrong_predictions.to_excel(
        writer,
        sheet_name="Wrong Predictions",
        index=False
    )


# ============================================================
# 12. PRINT REPORT
# ============================================================

print()
print("=" * 60)
print("DOCUMENT CLASSIFICATION REPORT")
print("=" * 60)

print()

print(f"Total Documents : {total}")

print(f"Correct         : {correct}")

print(f"Wrong           : {wrong}")

print(f"Errors          : {errors}")

print()

print(
    f"Accuracy        : {accuracy * 100:.2f}%"
)

print(
    f"Precision       : {precision * 100:.2f}%"
)

print(
    f"Recall          : {recall * 100:.2f}%"
)

print(
    f"F1 Score        : {f1 * 100:.2f}%"
)

print()

print("=" * 60)

print(
    f"Excel report created: {output_file}"
)

print("=" * 60)

print()