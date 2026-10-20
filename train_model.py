
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "student_results.csv"
MODEL_PATH = ROOT / "models" / "student_result_model.joblib"
REPORT_PATH = ROOT / "models" / "evaluation_report.txt"


# --------------------------------------------------
# TRAIN AND EVALUATE MODEL
# --------------------------------------------------

def main():
    # 1. Check dataset
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {"study_hours", "attendance", "passed"}

    if not required.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {sorted(required)}"
        )

    # Keep required columns and remove missing/invalid rows
    df = df[["study_hours", "attendance", "passed"]].copy()

    df["study_hours"] = pd.to_numeric(
        df["study_hours"], errors="coerce"
    )
    df["attendance"] = pd.to_numeric(
        df["attendance"], errors="coerce"
    )
    df["passed"] = pd.to_numeric(
        df["passed"], errors="coerce"
    )

    df = df.dropna()

    df = df[
        df["study_hours"].between(0, 24)
        & df["attendance"].between(0, 100)
        & df["passed"].isin([0, 1])
    ]

    df["passed"] = df["passed"].astype(int)

    if len(df) < 10:
        raise ValueError(
            "At least 10 valid rows are required for this demo evaluation."
        )

    if df["passed"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both classes: passed=0 and passed=1."
        )

    X = df[["study_hours", "attendance"]]
    y = df["passed"]

    print("=" * 55)
    print("STUDENT RESULT PREDICTION - MODEL EVALUATION")
    print("=" * 55)
    print(f"Valid dataset rows: {len(df)}")
    print(f"Pass examples: {(y == 1).sum()}")
    print(f"Needs-improvement examples: {(y == 0).sum()}")

    # 2. Split into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 3. Create the model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=42)),
    ])

    # 4. Train only on training data
    model.fit(X_train, y_train)

    # 5. Evaluate on the held-out test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    matrix = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Needs Improvement", "Pass"],
        zero_division=0,
    )

    # 6. Print evaluation results
    print("\n--- Test Metrics ---")
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    print("\n--- Confusion Matrix ---")
    print("Rows = actual class; columns = predicted class")
    print("Class order: 0 = Needs Improvement, 1 = Pass")
    print(matrix)

    print("\n--- Classification Report ---")
    print(report)

    # 7. Save the trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # 8. Save the evaluation report
    report_text = (
        "STUDENT RESULT PREDICTION - EVALUATION REPORT\n"
        "================================================\n"
        f"Valid dataset rows: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n"
        f"Precision: {precision:.4f}\n"
        f"Recall: {recall:.4f}\n"
        f"F1-score: {f1:.4f}\n\n"
        "Confusion Matrix (labels 0, 1):\n"
        f"{matrix}\n\n"
        "Classification Report:\n"
        f"{report}\n"
        "NOTE: Bundled sample data is synthetic and intended "
        "for educational use only.\n"
    )

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Evaluation report saved to: {REPORT_PATH}")
    print("\nTraining and evaluation completed.")
    print(
        "Note: bundled sample data is synthetic and intended "
        "for learning only."
    )


if __name__ == "__main__":
    main()
