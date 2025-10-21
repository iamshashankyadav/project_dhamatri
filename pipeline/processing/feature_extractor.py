import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Optional

# --- Global Model Initialization ---
# Initialize MediaPipe Pose for API performance.
try:
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5
    )
    logging.info("MediaPipe Pose Detector initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize MediaPipe Pose: {e}")
    pose_detector = None

# --- Feature Column Definition ---
# The final 17 features your SVR model expects
FEATURE_COLUMNS = [
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

def _dist_xy(a, b) -> float:
    """Euclidean distance in (x,y) normalized coordinates."""
    return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))


def _extract_per_image_measures(image_bytes: bytes) -> Optional[Dict[str, float]]:
    """
    Helper function to extract 9 base measures from a *single* image.
    This is identical to the 'extract_per_image_measures' from your notebook.
    """
    if not pose_detector:
        logging.warning("Pose detector is not available.")
        return None

    # 1. Decode image
    try:
        file_bytes = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            return None
    except Exception:
        return None

    # 2. Process image
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        res = pose_detector.process(image_rgb)
    except Exception as e:
        logging.error(f"Error during MediaPipe pose processing: {e}")
        return None
        
    if not res.pose_landmarks:
        return None  # No pose detected in this image

    lm = res.pose_landmarks.landmark
    LI = mp_pose.PoseLandmark

    # 3. Calculate 9 base measures
    try:
        L_SHO, R_SHO = lm[LI.LEFT_SHOULDER.value], lm[LI.RIGHT_SHOULDER.value]
        L_HIP, R_HIP = lm[LI.LEFT_HIP.value], lm[LI.RIGHT_HIP.value]
        L_WRIST, R_WRIST = lm[LI.LEFT_WRIST.value], lm[LI.RIGHT_WRIST.value]
        L_ANK, R_ANK = lm[LI.LEFT_ANKLE.value], lm[LI.RIGHT_ANKLE.value]
        NOSE = lm[LI.NOSE.value]
        L_EAR, R_EAR = lm[LI.LEFT_EAR.value], lm[LI.RIGHT_EAR.value]
        L_EYE_OUT, R_EYE_OUT = lm[LI.LEFT_EYE_OUTER.value], lm[LI.RIGHT_EYE_OUTER.value]

        mid_shoulder = ((L_SHO.x + R_SHO.x) / 2.0, (L_SHO.y + R_SHO.y) / 2.0)
        mid_hip = ((L_HIP.x + R_HIP.x) / 2.0, (L_HIP.y + R_HIP.y) / 2.0)

        shoulder_width = _dist_xy((L_SHO.x, L_SHO.y), (R_SHO.x, R_SHO.y))
        hip_width = _dist_xy((L_HIP.x, L_HIP.y), (R_HIP.x, R_HIP.y))
        torso_length = _dist_xy(mid_shoulder, mid_hip)
        head_to_shldr = _dist_xy((NOSE.x, NOSE.y), mid_shoulder)

        arm_L = _dist_xy((L_SHO.x, L_SHO.y), (L_WRIST.x, L_WRIST.y))
        arm_R = _dist_xy((R_SHO.x, R_SHO.y), (R_WRIST.x, R_WRIST.y))
        arm_len_avg = (arm_L + arm_R) / 2.0
        arm_len_max = max(arm_L, arm_R)

        leg_L = _dist_xy((L_HIP.x, L_HIP.y), (L_ANK.x, L_ANK.y))
        leg_R = _dist_xy((R_HIP.x, R_HIP.y), (R_ANK.x, R_ANK.y))
        leg_len_avg = (leg_L + leg_R) / 2.0
        leg_len_max = max(leg_L, leg_R)

        head_w_ears = _dist_xy((L_EAR.x, L_EAR.y), (R_EAR.x, R_EAR.y))
        head_w_eyes = _dist_xy((L_EYE_OUT.x, L_EYE_OUT.y), (R_EYE_OUT.x, R_EYE_OUT.y))
        head_width = head_w_ears if head_w_ears > 1e-6 else head_w_eyes

        return {
            "torso_length": torso_length,
            "leg_length_avg": leg_len_avg,
            "leg_length_max": leg_len_max,
            "arm_length_avg": arm_len_avg,
            "arm_length_max": arm_len_max,
            "head_to_shoulder": head_to_shldr,
            "shoulder_width": shoulder_width,
            "hip_width": hip_width,
            "head_width": head_width,
        }
    except Exception as e:
        logging.warning(f"Error calculating distances from landmarks: {e}. Skipping image.")
        return None


def aggregate_features_from_images(image_bytes_list: List[bytes]) -> Optional[Dict[str, float]]:
    """
    Main API function.
    Processes a list of 4 images, extracts per-image measures,
    and then aggregates them into the final 17 features.
    
    This function *exactly* replicates the training script's aggregation loop.
    """
    
    measures_per_image = []
    
    # 1. Loop through all 4 images and get the 9 base measures from each
    for img_bytes in image_bytes_list:
        m = _extract_per_image_measures(img_bytes)
        if m is not None:
            measures_per_image.append(m)

    # 2. Check if we found a pose in *any* image
    if not measures_per_image:
        logging.warning("No pose detected in *any* of the 4 images. Cannot extract features.")
        return None

    # 3. Create DataFrame (just like the training script)
    df_i = pd.DataFrame(measures_per_image)

    # 4. Perform aggregation (copied *exactly* from your training script)
    
    # 1) torso_length → max_torso_length, avg_torso_length
    max_torso_length = df_i["torso_length"].max()
    avg_torso_length = df_i["torso_length"].mean()

    # 2) leg_length → max_leg_length, avg_leg_length
    max_leg_length = df_i["leg_length_max"].max()
    avg_leg_length = df_i["leg_length_avg"].mean()

    # 3) arm_length → max_arm_length, avg_arm_length
    max_arm_length = df_i["arm_length_max"].max()
    avg_arm_length = df_i["arm_length_avg"].mean()

    # 4) head_to_shoulder → max_head_to_shoulder, avg_head_to_shoulder
    max_head_to_shoulder = df_i["head_to_shoulder"].max()
    avg_head_to_shoulder = df_i["head_to_shoulder"].mean()

    # 5) shoulder_width → max_shoulder_width, avg_shoulder_width
    max_shoulder_width = df_i["shoulder_width"].max()
    avg_shoulder_width = df_i["shoulder_width"].mean()

    # 6) hip_width → max_hip_width, avg_hip_width
    max_hip_width = df_i["hip_width"].max()
    avg_hip_width = df_i["hip_width"].mean()

    # 7) head_width → max_head_width, avg_head_width
    max_head_width = df_i["head_width"].max()
    avg_head_width = df_i["head_width"].mean()

    # 5. Calculate Ratios (also from training script)
    eps = 1e-8
    hip_to_shoulder = float(avg_hip_width / (avg_shoulder_width + eps))
    torso_to_leg = float(avg_torso_length / (avg_leg_length + eps))
    arm_to_torso = float(avg_arm_length / (avg_torso_length + eps))

    # 6. Build the final 17-feature dictionary
    final_features = {
        "max_torso_length": max_torso_length,
        "avg_torso_length": avg_torso_length,
        "max_leg_length": max_leg_length,
        "avg_leg_length": avg_leg_length,
        "max_arm_length": max_arm_length,
        "avg_arm_length": avg_arm_length,
        "max_head_to_shoulder": max_head_to_shoulder,
        "avg_head_to_shoulder": avg_head_to_shoulder,
        "max_shoulder_width": max_shoulder_width,
        "avg_shoulder_width": avg_shoulder_width,
        "max_hip_width": max_hip_width,
        "avg_hip_width": avg_hip_width,
        "max_head_width": max_head_width,
        "avg_head_width": avg_head_width,
        "hip_to_shoulder": hip_to_shoulder,
        "torso_to_leg": torso_to_leg,
        "arm_to_torso": arm_to_torso,
    }

    # # Final check
    # if set(final_features.keys()) != set(FEATURE_COLUMNS):
    #     logging.error("CRITICAL: Mismatch between generated features and required columns.")
    #     return None

    return final_features