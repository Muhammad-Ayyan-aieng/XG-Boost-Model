# api/routes/__init__.py
"""
Routes package - exports all route handlers
"""

from routes.health import router as health_router
from routes.predict import router as predict_router

# List of all routers for easy import in main.py
routers = [health_router, predict_router]

__all__ = ["health_router", "predict_router", "routers"]