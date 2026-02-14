# Smart Ambulance - Anomaly Detection & Triage System Documentation
  
**Module:** `anomaly.py`  
**Version:** 2.0 (Integrated Deep Learning & Intelligent Triage)

---

## 1. Executive Summary

This document outlines the technical design and logic behind the **Smart Ambulance Anomaly Detection System**. The system is designed to monitor patient vitals during transport, detecting life-threatening events while minimizing false alarms caused by ambulance motion (vibration).

The core innovation is the **Intelligent Triage Logic**, which combines:
1.  **Deep Learning (LSTM Autoencoder)**: To detect subtle, multivariate anomalies that standard thresholds might miss.
2.  **Clinical Risk Scoring**: Heuristic rules based on medical standards (Shock Index, SpO2 trends).
3.  **Motion-Aware Confidence Scoring**: A mechanism to suppress alerts when data quality is compromised by rough road conditions.

---

## 2. System Architecture

The `anomaly.py` script processes patient data through a four-stage pipeline:

### Stage 1: Data Ingestion & Preprocessing
*   **Input**: `smart_ambulance_synthetic_data.csv`
*   **Cleaning**: Maps raw sensor columns (`heart_rate`, `spo2`) to standardized names. Handles missing values using backward filling to preserve trend integrity.
*   **Scaling**: Standardizes data (mean=0, std=1) for optimal Deep Learning performance.

### Stage 2: Feature Engineering
We derive analytical features to capture patient dynamics, not just static values:
*   **Rolling Statistics (30s window)**: Mean and Standard Deviation for HR and SpO2.
*   **Trends**: Rate of change over time (e.g., rapidly dropping SpO2).
*   **Shock Index**: Calculated as `Heart Rate / Systolic BP`. A critical indicator of trauma/sepsis (Normal: 0.5-0.7, Critical: >0.9).

### Stage 3: Deep Learning Anomaly Detection
*   **Model**: LSTM Autoencoder (Long Short-Term Memory).
*   **Logic**: The model effectively "learns" what normal physiological patterns look like. It tries to reconstruct the input data.
*   **Anomaly Metric**: **Reconstruction Error (MSE)**. High error means the model is "surprised" by the data, indicating a potential anomaly (e.g., a sudden cardiac event).
*   **Threshold**: Dynamic threshold set at the 95th percentile of errors.

---

## 3. The New Triage & Risk Score Design

To address the high rate of false alarms in moving ambulances, we implemented a state-based triage system.

### A. The Components

1.  **Clinical Risk Score (0-100)**:
    *   Aggregates risk from Heart Rate (High/Low), SpO2 (Hypoxia), and Shock Index.
    *   **Threshold**: Score > 40 is considered "High Risk".

2.  **Confidence Score (0.0 - 1.0)**:
    *   Derived from the accelerometer (`vibration`) sensor.
    *   **Formula**: $Confidence = 1.0 - (Vibration / Max\_Vibration)$
    *   **Logic**: High vibration = Low Confidence. We assume sensor data is unreliable when the ambulance is shaking violently.

3.  **Persistence Window (5 seconds)**:
    *   A high-risk state must persist for at least 5 continual seconds to trigger a critical alert. This filters out momentary sensor glitches.

### B. Triage Status Classification

Every data point is classified into one of four statuses:

| Status | Logic | Interpretation | Action |
| :--- | :--- | :--- | :--- |
| **NORMAL** | Risk Score $\le$ 40 | Patient is stable. | **None** |
| **SUPPRESSED** (Low Confidence) | Risk > 40 **AND** Confidence < 0.5 | Vitals look bad, but the ambulance is shaking too much. | **Alert Suppressed** (Prevents False Positive) |
| **WARNING** (Transient) | Risk > 40 **AND** Confidence > 0.5 (Duration < 5s) | Valid high-risk reading, but too short to confirm. | **Monitor Closely** |
| **CRITICAL ALERT** | Risk > 40 **AND** Confidence > 0.5 (Duration $\ge$ 5s) | **Sustained, reliable high-risk event.** | **TRIGGER ALARM** |

---

## 4. Operational Results & Interpretation

### Outputs
1.  **`smart_ambulance_risk_scored_dl.csv`**: The processed dataset containing:
    *   `risk_score`: The calculated clinical risk.
    *   `confidence_score`: Data reliability metric.
    *   `triage_status`: The final decision classification.
    *   `reconstruction_error`: The deep learning anomaly metric.
2.  **`anomaly_risk_analysis_dl.png`**: A visualization dashboard.
    *   **Top Panel**: Vitals (HR, SpO2) with Deep Learning anomalies marked in black.
    *   **Bottom Panel**: Triage decisions. **Red Triangles** indicate Critical Alerts, while **Gray 'X's** show where alerts were safely suppressed due to vibration.
3.  **`lstm_autoencoder.pth`**: The trained PyTorch model state.
4.  **`scaler.pkl`**: The scaler object for processing future live data.

### Impact
This design ensures that:
*   **True emergencies** (e.g., patient crashing on a smooth highway) are caught immediately (`CRITICAL ALERT`).
*   **False alarms** (e.g., sensor noise while driving off-road) are filtered out (`SUPPRESSED`).
*   **The Board** can see a quantifiable reduction in "alarm fatigue" metrics.

---

## 5. Docker Setup

### Building the Docker Image
```bash
# Build the Docker image
docker build -t smart-ambulance-anomaly-detection .
```

### Running the Docker Container
```bash
# Run the Docker container
docker run -p 8000:8000 smart-ambulance-anomaly-detection
```

### Accessing the API
Once the container is running, you can access the API at `http://localhost:8000`.



**Prepared By:** Muhammed Nishad  

