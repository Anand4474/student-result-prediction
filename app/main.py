
from pathlib import Path
from datetime import datetime
import csv

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "student_result_model.joblib"
STATIC_DIR = ROOT / "app" / "static"
HTML_PATH = STATIC_DIR / "index.html"
DATA_DIR = ROOT / "data"
HISTORY_PATH = DATA_DIR / "prediction_history.csv"

app = FastAPI(
    title="Student Result Prediction API",
    description="Predict student results using study hours and attendance.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance: float = Field(ge=0, le=100)


def load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Model file not found: {MODEL_PATH}",
        )
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load model: {exc}",
        )


def read_history():
    if not HISTORY_PATH.is_file():
        return []

    try:
        df = pd.read_csv(HISTORY_PATH).fillna("")
        return df.to_dict(orient="records")
    except Exception:
        return []


def save_prediction(
    study_hours: float,
    attendance: float,
    prediction: str,
    passed: bool,
    pass_probability: float,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    columns = [
        "timestamp",
        "study_hours",
        "attendance",
        "prediction",
        "passed",
        "pass_probability",
    ]
    file_exists = HISTORY_PATH.exists()

    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists or HISTORY_PATH.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "study_hours": study_hours,
            "attendance": attendance,
            "prediction": prediction,
            "passed": passed,
            "pass_probability": round(pass_probability, 4),
        })


@app.get("/")
def home():
    return {
        "name": "Student Result Prediction API",
        "status": "running",
        "dashboard": "/ui",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_available": MODEL_PATH.is_file(),
        "frontend_exists": HTML_PATH.is_file(),
        "history_exists": HISTORY_PATH.is_file(),
    }


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    if not HTML_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Dashboard HTML not found: {HTML_PATH}",
        )

    html = HTML_PATH.read_text(encoding="utf-8-sig")
    if not html.strip():
        raise HTTPException(status_code=500, detail="index.html is empty.")

    return HTMLResponse(content=html)


@app.post("/predict")
def predict(data: PredictionInput):
    model = load_model()
    features = pd.DataFrame([{
        "study_hours": data.study_hours,
        "attendance": data.attendance,
    }])

    try:
        predicted_value = model.predict(features)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed. Check that the model was trained "
                f"with study_hours and attendance features. Details: {exc}"
            ),
        )

    prediction_text = str(predicted_value).strip()
    normalized = prediction_text.lower()

    if normalized in {"1", "true", "pass", "passed", "yes"}:
        passed = True
    elif normalized in {"0", "false", "fail", "failed", "no"}:
        passed = False
    else:
        # For models with labels such as "Needs Improvement", treat that
        # label as not passed. Adjust here if your training labels differ.
        passed = normalized in {"good", "excellent", "success"}

    pass_probability = 1.0 if passed else 0.0

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            pass_class_index = next(
                (
                    i for i, label in enumerate(classes)
                    if str(label).strip().lower()
                    in {"1", "true", "pass", "passed", "yes"}
                ),
                None,
            )
            if pass_class_index is not None:
                pass_probability = float(probabilities[pass_class_index])
            else:
                pass_probability = float(max(probabilities))
                # If class names are 0 and 1, index of class 1 is the pass
                # probability. For custom labels, review your class names.
                if not passed:
                    pass_probability = 1.0 - pass_probability
        except Exception:
            pass

    try:
        save_prediction(
            study_hours=data.study_hours,
            attendance=data.attendance,
            prediction=prediction_text,
            passed=passed,
            pass_probability=pass_probability,
        )
        history_saved = True
    except Exception:
        history_saved = False

    return {
        "prediction": prediction_text,
        "passed": passed,
        "pass_probability": round(pass_probability, 4),
        "input": {
            "study_hours": data.study_hours,
            "attendance": data.attendance,
        },
        "history_saved": history_saved,
        "note": "This is a model prediction, not a guarantee of the actual result.",
    }


@app.get("/history")
def history():
    records = read_history()

    pass_count = sum(
        str(record.get("passed", "")).strip().lower() in {"true", "1", "yes"}
        for record in records
    )

    # Latest records first
    records = list(reversed(records))

    return {
        "total": len(records),
        "pass_count": pass_count,
        "needs_improvement_count": len(records) - pass_count,
        "records": records[:100],
    }


@app.get("/history/export")
def export_history():
    if not HISTORY_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="No prediction history exists yet. Make a prediction first.",
        )

    return FileResponse(
        path=HISTORY_PATH,
        media_type="text/csv",
        filename="student_prediction_history.csv",
    )


@app.delete("/history")
def clear_history():
    if HISTORY_PATH.exists():
        HISTORY_PATH.unlink()

    return {"message": "Prediction history cleared."}

