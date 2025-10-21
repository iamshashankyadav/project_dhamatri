import cv2
import mediapipe as mp
import numpy as np
import logging

# --- Global Model Initialization ---
# This is a crucial optimization for a FastAPI server.
# The models are loaded once when the application starts,
# not on every single API request.

try:
    # 1. Initialize MediaPipe solutions
    mp_face_detection = mp.solutions.face_detection
    mp_pose = mp.solutions.pose

    # 2. Configure and create the model instances
    # We don't use 'with' statements here, as these instances
    # should persist for the application's entire lifetime.
    
    face_detector = mp_face_detection.FaceDetection(
        model_selection=1, 
        min_detection_confidence=0.5
    )

    pose_detector = mp_pose.Pose(
        static_image_mode=True,  # True because we process static images
        min_detection_confidence=0.5
    )
    
    logging.info("MediaPipe Human Detectors initialized successfully.")

except Exception as e:
    logging.error(f"Failed to initialize MediaPipe models: {e}")
    # If models fail to load, we set them to None so the function
    # can gracefully fail without crashing the app.
    face_detector = None
    pose_detector = None

# --- End Initialization ---


def is_human_present(image_bytes: bytes) -> bool:
    """
    Checks if a human is present in the image using MediaPipe.
    
    This function uses both Face Detection and Pose Estimation for
    a more robust check (e.g., detects a person from the back).

    Args:
        image_bytes: The raw bytes of the image file.

    Returns:
        True if a human face or pose is detected, False otherwise.
    """
    
    # Fail gracefully if models failed to load on startup
    if not face_detector or not pose_detector:
        logging.warning("Human detectors are not available. Skipping check.")
        # Depending on security policy, you might want to return False here.
        # For this use case, we'll assume it's okay to proceed.
        # Let's return False to enforce the check.
        return False

    # 1. Decode image from bytes
    try:
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            logging.warning("Failed to decode image. It may be corrupt.")
            return False
            
    except Exception as e:
        logging.error(f"Error decoding image bytes: {e}")
        return False

    # 2. Convert image to RGB (MediaPipe expects RGB format)
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        logging.error(f"OpenCV error during color conversion: {e}")
        return False

    # 3. Process the image with both models
    results_face = face_detector.process(image_rgb)
    results_pose = pose_detector.process(image_rgb)

    # 4. Return True if *either* model found a result
    # A face was detected OR pose landmarks were detected
    human_detected = bool(results_face.detections or results_pose.pose_landmarks)

    return human_detected