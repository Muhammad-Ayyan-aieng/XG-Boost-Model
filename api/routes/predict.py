# api/routes/predict.py
"""
Prediction endpoint - receives accident data, returns severity prediction
"""

from fastapi import APIRouter, HTTPException
from model_loader import model_loader
from utils.feature_builder import build_features
from models.request_response import AccidentInput, AccidentResponse

router = APIRouter()


def get_confidence(probability: float) -> str:
    """Convert probability to confidence level"""
    if probability >= 0.8:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.4:
        return "Medium"
    else:
        return "Low"


def get_severity_label(severity: int) -> str:
    """Convert severity number to label"""
    labels = {
        1: "Minor (Level 1)",
        2: "Moderate (Level 2)",
        3: "Serious (Level 3)",
        4: "Severe (Level 4)"
    }
    return labels.get(severity, "Unknown")


@router.post("/predict", response_model=AccidentResponse)
def predict(accident: AccidentInput):
    """
    Predict accident severity based on weather, time, road features, and keywords.
    
    Returns:
        - severity: 1-4 (1=Minor, 2=Moderate, 3=Serious, 4=Severe)
        - severity_label: Human-readable description
        - probability: Confidence of prediction (0-1)
        - confidence: Very High/High/Medium/Low
        - all_probabilities: Probability for each severity level
    """
    # Check if model is loaded
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert user input to 43 features
        features = build_features(accident)
        
        # Get prediction (0,1,2,3) and convert to (1,2,3,4)
        prediction = model_loader.predict(features)
        severity = prediction + 1
        
        # Get probabilities for each class
        probabilities = model_loader.predict_proba(features)
        probability = float(max(probabilities))
        
        # Prepare response
        return AccidentResponse(
            severity=severity,
            severity_label=get_severity_label(severity),
            probability=round(probability, 4),
            confidence=get_confidence(probability),
            all_probabilities={
                "severity_1": round(float(probabilities[0]), 4),
                "severity_2": round(float(probabilities[1]), 4),
                "severity_3": round(float(probabilities[2]), 4),
                "severity_4": round(float(probabilities[3]), 4)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")