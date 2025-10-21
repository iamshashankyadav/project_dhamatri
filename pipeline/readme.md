# Child Height Predictor API 📏

![FastAPI](https://img.shields.io/badge/Built%20with-FastAPI-green.svg)

This project deploys a trained Support Vector Regressor (SVR) model as a high-performance REST API using FastAPI. The API predicts a child's height (in cm) by analyzing a set of four images, extracting 17 key geometric features using MediaPipe.

The API replicates the exact data processing pipeline used during model training, ensuring feature consistency.

## 🚀 API Endpoint

### `POST /predict_height/`

Accepts **four** image files and returns a JSON object with the predicted height.

* **Request (Multipart Form-Data):**
    * `files`: A list containing exactly 4 image files (e.g., `.jpg`, `.png`).
* **Success Response (200 OK):**
    ```json
    {
      "predicted_height_cm": 97.5
    }
    ```
* **Error Responses:**
    * `400 Bad Request`: If 4 images are not provided, no human is detected in any image, or no pose can be found.
    * `503 Service Unavailable`: If the SVR model failed to load on startup.

---

## 🏗️ Project Architecture

The service is structured to separate concerns: API logic, image processing, and model handling.

### File Structure
### Request Workflow

1.  **Orchestration (`main.py`):** Receives 4 images at `POST /predict_height/`.
2.  **Validation (`human_detector.py`):** Loops through all 4 images. If a human is not detected in *any* of them, the request is rejected (HTTP 400).
3.  **Feature Extraction (`feature_extractor.py`):**
    * Runs MediaPipe Pose estimation on all 4 valid images.
    * Collects 9 base measures (e.g., `torso_length`, `arm_length_max`) from each image.
    * Aggregates these measures (e.g., `avg_torso_length`, `max_arm_length`) into the final 17 features, exactly matching the training script.
4.  **Prediction (`prediction.py`):** The 17 features are passed to the `ModelHandler`, which formats them into a 2D array and feeds them to the loaded SVR model.
5.  **Response (`main.py`):** The resulting float (e.g., `97.5`) is formatted as JSON and returned to the user.

---

## ⚙️ Setup and Installation

### Prerequisites

* Python 3.9+
* Your trained model file: `final_svr_height_predictor.pkl`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-repo-url/height_predictor_api.git
    cd height_predictor_api
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Add your model:**
    Place your trained `final_svr_height_predictor.pkl` file inside the `model/` directory. The application will not start without it.

---

## 🏃 Running the API

Run the API locally using Uvicorn (an ASGI server):

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
