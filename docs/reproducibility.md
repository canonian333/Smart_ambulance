# Smart Ambulance AI - Deployment Guide

## 1. Project Structure
```
/
├── anomaly.py           # Training Pipeline & Model Definition
├── app.py               # FastAPI Service for Inference
├── evaluate_alerts.py   # Metrics & Failure Analysis
├── requirements.txt     # Python Dependencies
├── scaler.pkl           # Trained Scaler Artifact
├── lstm_autoencoder.pth # Trained Model Weights
└── smart_ambulance_synthetic_data.csv # Dataset
```

## 2. Setup Environment
1.  Install Python 3.9+.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Training the Model
Run the main script to train the LSTM Autoencoder and generate artifacts (`scaler.pkl`, `lstm_autoencoder.pth`):
```bash
python anomaly.py
```
*   This will also produce the `smart_ambulance_risk_scored_dl.csv` and analysis plots.

## 4. Running the API Service
Start the FastAPI server:
```bash
uvicorn app:app --reload
```
*   The API will be available at `http://127.0.0.1:8000`.
*   Interactive Docs (Swagger UI): `http://127.0.0.1:8000/docs`.

## 5. API Usage Example
Send a **POST** request to `/predict` with JSON payload:
```json
{
  "timestamp": "2026-02-14T10:00:00",
  "heart_rate": 115.0,
  "spo2": 94.0,
  "sbp": 130.0,
  "dbp": 85.0,
  "vibration": 0.2
}
```

**Response:**
```json
{
  "is_anomaly": true,
  "risk_score": 45.0,
  "confidence_score": 0.92,
  "triage_status": "WARNING",
  "message": "Processed. MSE: 0.7214"
}
```

## 6. Metrics & Evaluation
To see the Precision, Recall, and Failure Analysis report:
```bash
python evaluate_alerts.py
```
This script acts as the Quality Assurance gate before deployment.
