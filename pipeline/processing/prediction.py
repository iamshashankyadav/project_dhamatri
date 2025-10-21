import joblib
import numpy as np
import logging
from typing import Dict, List

# --- Feature Order ---
# This is now the *single source of truth* for the model's input.
# We hard-code the 18 features in the exact order the SVR was trained on.
# The try...except block has been REMOVED.

FEATURE_COLUMNS: List[str] = [
    "age_in_months",  # The 1st feature
    
    # The 17 geometric features
    "max_torso_length", "avg_torso_length",
    "max_leg_length", "avg_leg_length",
    "max_arm_length", "avg_arm_length",
    "max_head_to_shoulder", "avg_head_to_shoulder",
    "max_shoulder_width", "avg_shoulder_width",
    "max_hip_width", "avg_hip_width",
    "max_head_width", "avg_head_width",
    "hip_to_shoulder",
    "torso_to_leg",
    "arm_to_torso"
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
        # This will now *always* use the 18-feature list defined above
        self.feature_order: List[str] = FEATURE_COLUMNS
        
        try:
            self.model = joblib.load(model_path)
            logging.info(f"SVR model loaded successfully from {model_path}")
            
            # This log will now correctly state 18 features
            logging.info(f"Model expected {len(self.feature_order)} features.") 
            
        except FileNotFoundError:
            logging.error(f"CRITICAL: Model file not found at {model_path}")
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
        try:
            # This will now loop over the 18-item list
            ordered_features = [features_dict[col] for col in self.feature_order]
        except KeyError as e:
            # This error will fire if 'age_in_months' is missing
            logging.error(f"Prediction failed: Missing feature in input dictionary: {e}")
            raise ValueError(f"Missing required feature for prediction: {e}")

        # 2. Convert the list into a 2D NumPy array.
        #    This will now correctly be a (1, 18) shape array
        input_data = np.array([ordered_features])

        # 3. Run the prediction
        try:
            prediction = self.model.predict(input_data)
            
            # 4. Return the result
            return float(prediction[0])
            
        except Exception as e:
            logging.error(f"Model .predict() method failed: {e}")
            raise RuntimeError(f"Model prediction failed: {e}")