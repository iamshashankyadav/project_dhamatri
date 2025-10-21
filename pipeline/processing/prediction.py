import joblib
import numpy as np
import logging
from typing import Dict, List

# --- Feature Order ---
# This import is the *most critical* part of the file.
# It guarantees that the features are fed into the model
# in the exact same order it was trained on.

try:
    # Import the single source of truth for feature names
    from processing.feature_extractor import FEATURE_COLUMNS
except ImportError:
    logging.error("CRITICAL: Could not import FEATURE_COLUMNS from processing.feature_extractor.")
    # As a fallback, hardcode. But this is a sign of a structural error.
    FEATURE_COLUMNS = [
        "max_torso_length", "avg_torso_length",
        "max_leg_length", "avg_leg_length",
        "max_arm_length", "avg_arm_length",
        "max_head_to_shoulder", "avg_head_to_shoulder",
        "max_shoulder_width", "avg_shoulder_width",
        "max_hip_width", "avg_hip_width",
        "max_head_width", "avg_head_width",
        "hip_to_shoulder", "torso_to_leg", "arm_to_torso"
    ]


class ModelHandler:
    """
    A class to load the SVR model and handle predictions.
    
    This wrapper class ensures the model is:
    1. Loaded only once on application startup.
    2. Correctly formats the input data for prediction.
    """
    
    def __init__(self, model_path: str):
        """
        Initializes the handler by loading the SVR model.
        
        Args:
            model_path: The file path to the .pkl model file.
        """
        self.model = None
        self.feature_order: List[str] = FEATURE_COLUMNS
        
        try:
            self.model = joblib.load(model_path)
            logging.info(f"SVR model loaded successfully from {model_path}")
            logging.info(f"Model expected {len(self.feature_order)} features.")
            
        except FileNotFoundError:
            logging.error(f"CRITICAL: Model file not found at {model_path}")
            # The app will be in a failed state, which main.py will handle.
        except Exception as e:
            logging.error(f"CRITICAL: Error loading model from {model_path}: {e}")
            
    def predict(self, features_dict: Dict[str, float]) -> float:
        """
        Runs a prediction using the loaded SVR model.
        
        Args:
            features_dict: A dictionary of {feature_name: value}
                           as provided by the feature_extractor.

        Returns:
            The predicted height as a single float.
            
        Raises:
            RuntimeError: If the model is not loaded.
            ValueError: If a required feature is missing.
        """
        
        if self.model is None:
            logging.error("Prediction attempt failed: Model is not loaded.")
            raise RuntimeError("Model is not loaded. Check application logs for initialization errors.")

        # --- Feature Dictionary to Array Conversion ---
        # 1. Create a list of feature values, ordered *exactly*
        #    by self.feature_order.
        try:
            ordered_features = [features_dict[col] for col in self.feature_order]
        except KeyError as e:
            logging.error(f"Prediction failed: Missing feature in input dictionary: {e}")
            raise ValueError(f"Missing required feature for prediction: {e}")

        # 2. Convert the list into a 2D NumPy array.
        #    Scikit-learn models expect a 2D array: [[f1, f2, ..., f17]]
        input_data = np.array([ordered_features])

        # 3. Run the prediction
        try:
            prediction = self.model.predict(input_data)
            
            # 4. Return the result as a single float
            #    .predict() returns an array (e.g., np.array([97.5])),
            #    so we extract the first (and only) element.
            return float(prediction[0])
            
        except Exception as e:
            logging.error(f"Model .predict() method failed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")