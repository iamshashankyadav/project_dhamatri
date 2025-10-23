import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request,Form
from typing import List
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Local Module Imports ---
# Import the logic from our other project files.
from processing.prediction import ModelHandler ,WeightModelHandler
from processing.human_detector import is_human_present
from processing.feature_extractor import aggregate_features_from_images

# --- Configuration ---
MODEL_PATH = "model/final_svr_height_predictor_with_gender.pkl"
WEIGHT_ARTIFACT_PATH = "model/weight_model_artifacts.pkl"
EXPECTED_IMAGE_COUNT = 4  # As per your multi-image training logic

# --- Logging ---
# Set up basic logging to see app activity
logging.basicConfig(level=logging.INFO)
# Use the Uvicorn logger
log = logging.getLogger("uvicorn.error")

# --- Model Loading (Lifespan Event) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's startup and shutdown events.
    Loads the ML model on startup and stores it in app.state.
    """
    log.info("Application starting up...")
    
    # 1. Load the SVR model
    model_handler_instance = ModelHandler(MODEL_PATH)
    
    # 2. Check if model loaded successfully
    if model_handler_instance.model is None:
        log.critical(f"Model from {MODEL_PATH} failed to load. API will be non-functional.")
        # Store None to indicate failure
        app.state.model_handler = None
    else:
        log.info("SVR model loaded successfully.")
        # 3. Store the handler in the app's state
        app.state.model_handler = model_handler_instance

    # 2. Load the weight model artifacts (scaler + linear model)
    weight_handler_instance = WeightModelHandler(WEIGHT_ARTIFACT_PATH)
    if weight_handler_instance.model is None or getattr(weight_handler_instance, "scaler", None) is None:
        log.critical(f"Weight model artifacts from {WEIGHT_ARTIFACT_PATH} failed to load.")
        app.state.weight_handler = None
    else:
        log.info("Weight model loaded successfully.")
        app.state.weight_handler = weight_handler_instance
    
    yield
    
    # --- Shutdown ---
    log.info("Application shutting down...")
    app.state.model_handler = None
    app.state.weight_handler = None


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Child Height Predictor API",
    description=f"Predicts child height from {EXPECTED_IMAGE_COUNT} images using an SVR model.",
    version="1.0.0",
    lifespan=lifespan # Use the lifespan manager
)


# --- Pydantic Response Model ---
class PredictionResponse(BaseModel):
    """
    Defines the JSON structure for a successful prediction.
    """
    predicted_height_cm: float

class WeightResponse(BaseModel):
    """
    Response model for weight prediction.
    """
    predicted_weight_grams: float


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(request: Request):
    """
    Simple health check to verify the app is running and the models are loaded.
    """
    model_handler = getattr(request.app.state, "model_handler", None)
    weight_handler = getattr(request.app.state, "weight_handler", None)


    height_ok = bool(model_handler and getattr(model_handler, "model", None))
    weight_ok = bool(weight_handler and getattr(weight_handler, "model", None) and getattr(weight_handler, "scaler", None))


    if height_ok and weight_ok:
        return {"status": "ok", "height_model_loaded": True, "weight_model_loaded": True}


    log.warning("Health check failed: One or more models not loaded.")
    return {"status": "error", "height_model_loaded": height_ok, "weight_model_loaded": weight_ok}

# --- Prediction Endpoint (Updated for 19 Features) ---
@app.post("/predict_height/", response_model=PredictionResponse)
async def predict_height_from_images(
    request: Request,
    
    files: List[UploadFile] = File(
        ..., 
        description=f"A list of exactly {EXPECTED_IMAGE_COUNT} images of the child."
    ),
    age_in_months: float = Form(
        ..., 
        gt=0,
        description="The child's age in months."
    ),
    # --- ADDED GENDER FIELD ---
    gender: str = Form(
        ...,
        description="The child's gender: 'm' (male) or 'f' (female)."
    )
):
    """
    Orchestrates the 5-step prediction pipeline.
    """
    
    # --- Step 0: Check Model State ---
    model_handler = request.app.state.model_handler
    if not model_handler or not model_handler.model:
        log.error("Prediction failed: Model is not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please contact the administrator."
        )

    # --- Step 1: Orchestration & Input Validation ---
    
    # 1a. Check image count
    if len(files) != EXPECTED_IMAGE_COUNT:
        log.warning(f"Request rejected: Received {len(files)} images, expected {EXPECTED_IMAGE_COUNT}.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Exactly {EXPECTED_IMAGE_COUNT} image files are required."
        )

    # 1b. Read all image bytes
    image_bytes_list = []
    for file in files:
        if not file.content_type.startswith("image/"):
             raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file.filename}' is not a valid image type."
            )
        image_bytes_list.append(await file.read())

    # --- Step 2: Validation (Human Detector) ---
    log.info("Step 2: Validating images for human presence...")
    for i, img_bytes in enumerate(image_bytes_list):
        if not is_human_present(img_bytes):
            log.warning(f"Request rejected: No human detected in image {i+1} ('{files[i].filename}').")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No human was detected in image '{files[i].filename}'. Please upload 4 clear images."
            )
    
    log.info(f"All {EXPECTED_IMAGE_COUNT} images passed human detection.")

    # --- Step 3: Feature Extraction ---
    log.info("Step 3: Extracting features...")
    
    try:
        # 1. Get the 17 geometric features
        features_dict = aggregate_features_from_images(image_bytes_list)
        
    except Exception as e:
        log.error(f"An unexpected error occurred during feature extraction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during feature extraction: {e}"
        )

    # Check for pose detection failure
    if features_dict is None:
        log.warning("Request rejected: No pose could be detected in any of the 4 images.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not detect a pose in any of the images. Please use clearer, full-body photos."
        )
            
    # --- ADDED GENDER ENCODING ---
    # 2. Validate and encode 'gender'
    gender_norm = gender.lower().strip()
    if gender_norm == 'm':
        gender_encoded = 1
    elif gender_norm == 'f':
        gender_encoded = 0
    else:
        # If input is not 'm' or 'f', reject the request.
        log.warning(f"Request rejected: Invalid gender input. Received '{gender}'.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid gender value. Please use 'm' or 'f'."
        )

    # 3. Add the two label-encoded features
    features_dict["gender"] = gender_encoded
    features_dict["age_in_months"] = age_in_months
    
    log.info(f"Successfully prepared {len(features_dict)} features.") # Will now log 19

    # --- Step 4: Prediction ---
    log.info("Step 4: Running prediction...")
    try:
        predicted_height = model_handler.predict(features_dict)
    
    except (RuntimeError, ValueError) as e:
        log.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model prediction failed: {e}"
        )
    
    log.info(f"Prediction successful. Result: {predicted_height:.2f} cm")

    # --- Step 5: Format and Return Response ---
    return PredictionResponse(predicted_height_cm=predicted_height)

# --- Weight Prediction Endpoint (New) ---
@app.post("/predict_weight/", response_model=WeightResponse)
async def predict_weight_simple(
    request: Request,
    height_in_cm: float = Form(..., gt=0, description="Child's height in cm (predicted by height API or measured)."),
    age_in_months: float = Form(..., gt=0, description="Child's age in months."),
    gender: str = Form("m", description="Child's gender; defaults to 'm' if not provided.")
):
    """
    Simple, decoupled weight prediction:
    - All inputs are plain form fields.
    - No image processing or height inference here.
    """
    weight_handler = getattr(request.app.state, "weight_handler", None)
    if not weight_handler or weight_handler.model is None or getattr(weight_handler, "scaler", None) is None:
        log.error("Weight prediction failed: weight model not loaded.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Weight model is not loaded.")


    try:
        predicted_weight = weight_handler.predict(height_in_cm, age_in_months, gender)
    except Exception as e:
        log.error(f"Weight prediction error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Weight prediction failed: {e}")

    return WeightResponse(predicted_weight_grams=predicted_weight) 