import pandas as pd
from predictor import predict_dataset

print("========================================")
print(" NIDS MODEL VALIDATION")
print("========================================")

# --------------------------------------------------
# Load Real UNSW-NB15 Testing Dataset
# --------------------------------------------------

TEST_PATH = r"C:\NIDS_\data\raw\UNSW_NB15_testing-set.csv"

df = pd.read_csv(TEST_PATH)

print("\nTesting Dataset Loaded Successfully")

print("Dataset Shape:", df.shape)


# --------------------------------------------------
# Save Actual Labels
# BEFORE sending data to predictor
# --------------------------------------------------

actual_binary = df["label"].copy()
actual_attack_category = df["attack_cat"].copy()


# --------------------------------------------------
# Remove Labels
# Dashboard/predictor should NOT see the answers
# --------------------------------------------------

input_df = df.drop(
    ["label", "attack_cat"],
    axis=1
)


# --------------------------------------------------
# Run Prediction
# --------------------------------------------------

print("\nRunning NIDS Prediction...")

results = predict_dataset(input_df)

print("\nPrediction Completed Successfully")


# --------------------------------------------------
# Compare Binary Prediction
# --------------------------------------------------

predicted_binary = (
    results["Binary Prediction"]
    .map({
        "Normal": 0,
        "Attack": 1
    })
)


binary_correct = (
    predicted_binary == actual_binary
).sum()

total_records = len(actual_binary)

binary_accuracy = (
    binary_correct / total_records
) * 100


# --------------------------------------------------
# Binary Results
# --------------------------------------------------

print("\n========================================")
print(" BINARY CLASSIFICATION VALIDATION")
print("========================================")

print(
    "Correct Predictions:",
    binary_correct
)

print(
    "Total Records:",
    total_records
)

print(
    "Binary Accuracy:",
    round(binary_accuracy, 2),
    "%"
)


# --------------------------------------------------
# Binary Confusion Matrix
# --------------------------------------------------

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

binary_cm = confusion_matrix(
    actual_binary,
    predicted_binary
)

print("\nBinary Confusion Matrix")

print(binary_cm)


print("\nBinary Classification Report")

print(
    classification_report(
        actual_binary,
        predicted_binary,
        target_names=[
            "Normal",
            "Attack"
        ]
    )
)


# --------------------------------------------------
# Multi-Class Validation
# --------------------------------------------------

print("\n========================================")
print(" MULTI-CLASS VALIDATION")
print("========================================")


actual_attack_category = (
    actual_attack_category
    .astype(str)
)


predicted_attack_category = (
    results["Attack Category"]
    .astype(str)
)


multiclass_correct = (
    predicted_attack_category
    == actual_attack_category
).sum()


multiclass_accuracy = (
    multiclass_correct
    / total_records
) * 100


print(
    "Correct Predictions:",
    multiclass_correct
)

print(
    "Total Records:",
    total_records
)

print(
    "Multi-Class Accuracy:",
    round(
        multiclass_accuracy,
        2
    ),
    "%"
)


# --------------------------------------------------
# Multi-Class Classification Report
# --------------------------------------------------

print("\nMulti-Class Classification Report")

print(
    classification_report(
        actual_attack_category,
        predicted_attack_category
    )
)


# --------------------------------------------------
# Final Summary
# --------------------------------------------------

print("\n========================================")
print(" FINAL VALIDATION SUMMARY")
print("========================================")

print(
    f"Binary Accuracy: "
    f"{binary_accuracy:.2f}%"
)

print(
    f"Multi-Class Accuracy: "
    f"{multiclass_accuracy:.2f}%"
)

print("\nValidation Completed Successfully.")       