from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

# --- Model Definition (Must match training) ---
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        self.encoder1 = nn.LSTM(input_size=n_features, hidden_size=32, num_layers=1, batch_first=True)
        self.encoder2 = nn.LSTM(input_size=32, hidden_size=embedding_dim, num_layers=1, batch_first=True)
        self.decoder1 = nn.LSTM(input_size=embedding_dim, hidden_size=32, num_layers=1, batch_first=True)
        self.decoder2 = nn.LSTM(input_size=32, hidden_size=n_features, num_layers=1, batch_first=True)
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

    def forward(self, x):
        x, (_, _) = self.encoder1(x)
        x, (hidden_n, _) = self.encoder2(x)
        latent = hidden_n[-1]
        x = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (_, _) = self.decoder1(x)
        x, (_, _) = self.decoder2(x)
        return x

# --- App Setup ---
app = FastAPI(title="Smart Ambulance AI Service", description="Real-time Anomaly Detection & Clinical Risk Scoring")

# Global Variables for Artifacts
model = None
scaler = None
SEQ_LEN = 30
FEATURES = ['hr_cleaned', 'spo2_cleaned', 'sbp', 'hr_std', 'spo2_trend'] 
# We need a buffer to store the last 30 seconds of data for sequence generation
data_buffer = []

@app.on_event("startup")
def load_artifacts():
    global model, scaler
    try:
        # Load Scaler
        scaler = joblib.load("scaler.pkl")
        
        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=len(FEATURES)).to(device)
        model.load_state_dict(torch.load("lstm_autoencoder.pth", map_location=device))
        model.eval()
        print("Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

# --- API Definitions ---

class VitalsInput(BaseModel):
    timestamp: str
    heart_rate: float
    spo2: float
    sbp: float
    dbp: float
    vibration: float # 0.0 to 1.0

class RiskOutput(BaseModel):
    is_anomaly: bool
    risk_score: float
    confidence_score: float
    triage_status: str
    message: str

@app.post("/predict", response_model=RiskOutput)
def predict_risk(vitals: VitalsInput):
    global data_buffer
    
    # 1. Preprocess Input
    # Convert input to the feature set expected by the model
    # Note: We need historical data to calculate trends and rolling stats (hr_std, spo2_trend)
    # For this simplified API, we will APPEND to a global buffer.
    
    current_data = {
        'hr_cleaned': vitals.heart_rate,
        'spo2_cleaned': vitals.spo2,
        'sbp': vitals.sbp,
        'vibration': vitals.vibration
    }
    data_buffer.append(current_data)
    
    # Keep buffer size manageable (e.g., last 60 seconds)
    if len(data_buffer) > 60:
        data_buffer.pop(0)
        
    # Convert buffer to DataFrame for feature engineering
    df = pd.DataFrame(data_buffer)
    
    # Engineer Features (Rolling/Trends)
    # We need at least 2 points for diff, 30 for full window
    if len(df) < 2:
        return {
            "is_anomaly": False, "risk_score": 0, "confidence_score": 1.0, 
            "triage_status": "BUFFERING", "message": "Collecting more data..."
        }
        
    df['hr_mean'] = df['hr_cleaned'].rolling(window=30, min_periods=1).mean()
    df['hr_std'] = df['hr_cleaned'].rolling(window=30, min_periods=1).std().fillna(0)
    df['spo2_trend'] = df['spo2_cleaned'].diff().fillna(0) # Simple diff for now
    
    # Select the LATEST row for prediction
    latest_features = df.iloc[-1][FEATURES].values.reshape(1, -1)
    
    # 2. Deep Learning Inference (Anomaly Detection)
    # Need a sequence of length 30
    is_anomaly = False
    reconstruction_error = 0.0
    
    if len(df) >= SEQ_LEN:
        # Create sequence
        seq_data = df[FEATURES].iloc[-SEQ_LEN:].values
        # Scale
        if scaler:
            seq_scaled = scaler.transform(seq_data)
            # Tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                loss = nn.MSELoss(reduction='none')(output, input_tensor)
                reconstruction_error = loss.mean().item()
                
        # Threshold Check (Hardcoded from training or loaded)
        THRESHOLD_MSE = 0.60
        is_anomaly = reconstruction_error > THRESHOLD_MSE

    # 3. Clinical Risk Score Calculation
    # Re-implement the logic from anomaly.py
    risk_score = 0
    # HR Score
    hr = vitals.heart_rate
    hr_trend = df.iloc[-1]['hr_mean'] - df.iloc[-2]['hr_mean'] if len(df) > 1 else 0
    if hr > 120 or hr < 40: risk_score += 30
    elif hr > 100 or hr < 50: risk_score += 10
    if hr_trend > 0.5: risk_score += 5
    
    # SpO2 Score
    spo2 = vitals.spo2
    spo2_trend = df.iloc[-1]['spo2_trend']
    if spo2 < 90: risk_score += 40
    elif spo2 < 95: risk_score += 15
    if spo2_trend < -0.2: risk_score += 10
    
    # Shock Index
    si = hr / vitals.sbp if vitals.sbp > 0 else 0
    if si > 0.9: risk_score += 20
    
    risk_score = min(risk_score, 100)
    
    # 4. Confidence Score
    vib_max = 2.5 # Estimated max vibration
    confidence_score = max(0, min(1, 1.0 - (vitals.vibration / vib_max)))
    
    # 5. Triage Logic
    triage_status = "NORMAL"
    if risk_score > 40:
        if confidence_score < 0.5:
            triage_status = "SUPPRESSED (Low Confidence)"
        else:
            triage_status = "WARNING" # In API, we might return Warning immediately
            
            # Simple Persistence Check in API (Stateful)
            # In a real app, use Redis. Here, we check the buffer.
            # Count recent high risk frames
            high_risk_count = 0
            for i in range(min(5, len(df))):
                # Re-calc risk for past 5 frames? Expensive.
                # Simplified: If current is high risk and we have previous high risks
                pass 
            # For this demo, we mark as WARNING unless client tracks persistence
            
            if is_anomaly: 
                 triage_status = "CRITICAL ALERT (DL Confirmed)"

    return {
        "is_anomaly": is_anomaly,
        "risk_score": float(risk_score),
        "confidence_score": float(confidence_score),
        "triage_status": triage_status,
        "message": f"Processed. MSE: {reconstruction_error:.4f}"
    }

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)
