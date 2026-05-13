# api/routes/health.py
"""
Health check endpoint to verify API status
"""

from fastapi import APIRouter
from model_loader import model_loader

router = APIRouter()


@router.get("/health")
def health_check():
    """
    GET /health
    Returns API health status and model information
    """
    return {
        "status": "ok",
        "model_loaded": model_loader.is_loaded(),
        "model_version": "1.0.0"
    }


@router.get("/ready")
def readiness_check():
    """
    GET /ready
    Kubernetes readiness probe - checks if model is loaded
    """
    if model_loader.is_loaded():
        return {"status": "ready"}
    else:
        return {"status": "not ready"}, 503