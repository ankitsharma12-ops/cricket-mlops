from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

# ── App Setup ────────────────────────────────────────
app = FastAPI(
    title="Cricket Outcome Predictor API",
    description="Predicts IPL match outcome based on toss & team data",
    version="1.0.0"
)

# ── Load Model ───────────────────────────────────────
MODEL_PATH = "models/model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from {MODEL_PATH}")
    return model

model = load_model()

# ── Request Schema ────────────────────────────────────
class MatchInput(BaseModel):
    team1:         int
    team2:         int
    venue:         int
    toss_winner:   int
    toss_decision: int

    class Config:
        json_schema_extra = {
            "example": {
                "team1":         7,
                "team2":         0,
                "venue":         55,
                "toss_winner":   7,
                "toss_decision": 0
            }
        }

# ── Response Schema ───────────────────────────────────
class PredictionOutput(BaseModel):
    toss_win_match_win: int
    probability:        float
    interpretation:     str

# ── Routes ────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Cricket Outcome Predictor API",
        "docs":    "Visit /docs for Swagger UI",
        "health":  "/health"
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model":  "random_forest",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: MatchInput):
    try:
        # Build input DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Predict
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Human-readable interpretation
        if prediction == 1:
            interpretation = f"Toss winner is likely to WIN the match ({round(probability * 100, 1)}% confidence)"
        else:
            interpretation = f"Toss winner is likely to LOSE the match ({round((1 - probability) * 100, 1)}% confidence)"

        return PredictionOutput(
            toss_win_match_win=int(prediction),
            probability=round(float(probability), 4),
            interpretation=interpretation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))