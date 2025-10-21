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