# api/model_loader.py
"""
Loads and caches the trained XGBoost model for API predictions
"""

import joblib
import os
import numpy as np
import logging
from config import MODEL_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Singleton class that loads model once and serves predictions"""
    
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, model_path=None):
        """Load model from disk"""
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = MODEL_PATH
        
        print(f"🔍 LOADING MODEL FROM: {model_path}")

        try:
            self._model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = None
            return False
    
    def predict(self, features):
        """Return severity class (0,1,2,3)"""
        if self._model is None:
            raise ValueError("Model not loaded")
        
        features_array = np.array(features).reshape(1, -1)
        return self._model.predict(features_array)[0]
    
    def predict_proba(self, features):
        """Return probabilities for each class"""
        if self._model is None:
            raise ValueError("Model not loaded")
        
        features_array = np.array(features).reshape(1, -1)
        return self._model.predict_proba(features_array)[0]
    
    def is_loaded(self):
        return self._model is not None


model_loader = ModelLoader()