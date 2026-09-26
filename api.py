import os
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app_model import load_model, predict_probability
from llm import explain_prediction, chat_with_llm


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "model/hybrid_model.pth"
SCALER_PATH = "model/scaler.pkl"


FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",

    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",

    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="HEXAMIND Hybrid Quantum ML API",
    description=(
        "Hybrid quantum-classical machine-learning "
        "prediction system with AI explanation."
    ),
    version="2.1"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# CHECK MODEL FILES
# =========================================================

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        "hybrid_model.pth not found. "
        "Make sure the trained model exists."
    )


if not os.path.exists(SCALER_PATH):
    raise RuntimeError(
        "scaler.pkl not found. "
        "Make sure the scaler exists."
    )


# =========================================================
# LOAD MODEL ONCE
# =========================================================

model = load_model(MODEL_PATH)

scaler = joblib.load(
    SCALER_PATH
)


# =========================================================
# REQUEST MODELS
# =========================================================

class PredictionRequest(BaseModel):

    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30
    )


class ChatRequest(BaseModel):

    messages: list[dict]


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "system": "HEXAMIND",
        "model": "Hybrid Quantum-Classical QNN",
        "qubits": 4,
        "features": 30,
        "ai": "online",
        "version": "2.1"
    }


# =========================================================
# FEATURE CONTRIBUTION CALCULATION
# =========================================================

def calculate_feature_contributions(
    scaled_features
):

    # -----------------------------------------------------
    # Base probability
    # -----------------------------------------------------

    base_benign = float(
        predict_probability(
            model,
            scaled_features
        )
    )

    base_malignant = (
        1.0 - base_benign
    )


    contributions = []


    # -----------------------------------------------------
    # Sensitivity step
    # -----------------------------------------------------

    delta = 0.25


    # -----------------------------------------------------
    # Test every feature
    # -----------------------------------------------------

    for i, feature_name in enumerate(
        FEATURE_NAMES
    ):

        # ---------------------------------------------
        # Feature increased
        # ---------------------------------------------

        higher = scaled_features.copy()

        higher[0, i] += delta


        higher_benign = float(
            predict_probability(
                model,
                higher
            )
        )


        higher_malignant = (
            1.0 - higher_benign
        )


        # ---------------------------------------------
        # Feature decreased
        # ---------------------------------------------

        lower = scaled_features.copy()

        lower[0, i] -= delta


        lower_benign = float(
            predict_probability(
                model,
                lower
            )
        )


        lower_malignant = (
            1.0 - lower_benign
        )


        # ---------------------------------------------
        # Central sensitivity
        # ---------------------------------------------

        contribution = (
            higher_malignant
            - lower_malignant
        ) / (2.0 * delta)


        contribution = float(
            contribution
        )


        # ---------------------------------------------
        # Direction
        # ---------------------------------------------

        if contribution > 0.000001:

            direction = (
                "toward malignant"
            )

        elif contribution < -0.000001:

            direction = (
                "toward benign"
            )

        else:

            direction = (
                "negligible effect"
            )


        contributions.append(
            {
                "feature": feature_name,

                "contribution": round(
                    contribution,
                    8
                ),

                "direction": direction
            }
        )


    # -----------------------------------------------------
    # Sort by absolute influence
    # -----------------------------------------------------

    contributions.sort(
        key=lambda x: abs(
            x["contribution"]
        ),
        reverse=True
    )


    return contributions


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(
    request: PredictionRequest
):

    # -----------------------------------------------------
    # Convert input
    # -----------------------------------------------------

    raw_features = np.asarray(
        request.features,
        dtype=np.float32
    )


    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    if len(raw_features) != 30:

        raise HTTPException(
            status_code=400,
            detail=(
                "Exactly 30 medical features "
                "are required."
            )
        )


    # -----------------------------------------------------
    # Scale
    # -----------------------------------------------------

    scaled_features = scaler.transform(
        raw_features.reshape(1, -1)
    )


    # -----------------------------------------------------
    # Model prediction
    # -----------------------------------------------------

    probability_benign = float(
        predict_probability(
            model,
            scaled_features
        )
    )


    probability_malignant = (
        1.0 - probability_benign
    )


    # -----------------------------------------------------
    # Classification
    # -----------------------------------------------------

    if probability_benign >= 0.5:

        prediction = "Benign"

        confidence = (
            probability_benign
        )

    else:

        prediction = "Malignant"

        confidence = (
            probability_malignant
        )


    # =====================================================
    # FEATURE EXPLANATION
    # =====================================================

    try:

        contributions = (
            calculate_feature_contributions(
                scaled_features
            )
        )


        top_features = contributions[:5]


    except Exception as e:

        print(
            "Feature contribution error:",
            str(e)
        )

        top_features = []


    # =====================================================
    # AI EXPLANATION
    # =====================================================

    try:

        explanation = explain_prediction(

            prediction=prediction,

            confidence=round(
                confidence * 100,
                2
            ),

            malignant_probability=round(
                probability_malignant * 100,
                2
            ),

            benign_probability=round(
                probability_benign * 100,
                2
            ),

            top_features=top_features
        )


        llm_status = "success"


    except Exception as e:

        print(
            "LLM error:",
            str(e)
        )


        explanation = (
            "The Hybrid Quantum ML model "
            "generated a prediction successfully, "
            "but the AI explanation service is "
            "currently unavailable."
        )


        llm_status = "unavailable"


    # =====================================================
    # RESPONSE
    # =====================================================

    return {

        "prediction": prediction,

        "confidence": round(
            confidence * 100,
            2
        ),

        "malignant_probability": round(
            probability_malignant * 100,
            2
        ),

        "benign_probability": round(
            probability_benign * 100,
            2
        ),

        "top_features": top_features,

        "explanation": explanation,

        "llm_status": llm_status,

        "disclaimer": (
            "This is a machine-learning prediction "
            "for research and decision-support "
            "purposes. It is not a medical diagnosis."
        )
    }


# =========================================================
# GENERAL AI CHAT
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        reply = chat_with_llm(
            request.messages
        )


        return {
            "reply": reply,
            "status": "success"
        }


    except Exception as e:

        print(
            "Chat LLM error:",
            str(e)
        )


        raise HTTPException(
            status_code=500,
            detail=(
                "AI chat service is currently "
                "unavailable."
            )
        )