import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Define paths
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, "data", "smart_ambulance_synthetic_data.csv")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim=16):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder1 = nn.LSTM(
            input_size=n_features, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True
        )
        self.encoder2 = nn.LSTM(
            input_size=32, 
            hidden_size=embedding_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        # Decoder
        self.decoder1 = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=32, 
            num_layers=1, 
            batch_first=True
        )
        self.decoder2 = nn.LSTM(
            input_size=32, 
            hidden_size=n_features, 
            num_layers=1, 
            batch_first=True
        )
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x shape: (batch_size, seq_len, n_features)
        
        # Encoder
        x, (_, _) = self.encoder1(x)
        x, (hidden_n, _) = self.encoder2(x)
        
        # Latent representation (last hidden state of encoder)
        # hidden_n shape: (num_layers, batch_size, embedding_dim) -> (1, batch, 16)
        latent = hidden_n[-1] # Shape: (batch, 16)
        
        # Repeat vector to match sequence length for decoder
        x = latent.unsqueeze(1).repeat(1, self.seq_len, 1) # Shape: (batch, seq_len, 16)
        
        # Decoder
        x, (_, _) = self.decoder1(x)
        x, (_, _) = self.decoder2(x) # Output: (batch, seq_len, n_features)
        
        return x

def create_sequences(data, seq_len):
    xs = []
    for i in range(len(data) - seq_len):
        x = data[i:(i + seq_len)]
        xs.append(x)
    return np.array(xs)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Map columns if they don't exist (handle synthetic data column names)
        if 'hr_cleaned' not in df.columns and 'heart_rate' in df.columns:
             df['hr_cleaned'] = df['heart_rate']
        if 'spo2_cleaned' not in df.columns and 'spo2' in df.columns:
             df['spo2_cleaned'] = df['spo2']
             
        return df
    except FileNotFoundError:
        print("Cleaned data file not found.")
        return None

from sklearn.model_selection import train_test_split

# ... imports ...

def train_model(model, train_loader, val_loader, epochs=20, device='cpu', patience=5):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training LSTM Autoencoder for {epochs} epochs with Early Stopping...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for data in train_loader:
            inputs = data[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_loss += loss.item() * inputs.size(0)
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
            
    return model

def get_reconstruction_error(model, data_loader, device='cpu'):
    model.eval()
    losses = []
    criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            outputs = model(inputs)
            # Loss per sample: mean over features and sequence interval
            # shape: (batch, seq, features)
            loss = criterion(outputs, inputs)
            # Mean error per sample
            loss = loss.mean(dim=[1, 2])
            losses.extend(loss.cpu().numpy())
            
    return np.array(losses)

# ... (rest of file) ...

def run_deep_anomaly_detection(df):
    """
    Task 2A: Deep Learning Anomaly Detection (LSTM Autoencoder).
    """
    print("Preparing data for Deep Learning model...")
    
    # Select features
    features = ['hr_cleaned', 'spo2_cleaned', 'sbp', 'hr_std', 'spo2_trend'] 
    
    # Standardize
    scaler = StandardScaler()
    data = df[features].bfill().fillna(0).values
    data_scaled = scaler.fit_transform(data)
    
    # Save Scaler for later use (Inference)
    scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Create sequences
    SEQ_LEN = 30
    X_sequences = create_sequences(data_scaled, SEQ_LEN)
    
    # Split into Train and Validation sets (80/20)
    X_train, X_val = train_test_split(X_sequences, test_size=0.2, random_state=42)
    
    # PyTorch Data Loaders
    # Train Loader (for optimized training)
    train_dataset = TensorDataset(torch.Tensor(X_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Val Loader (for early stopping)
    val_dataset = TensorDataset(torch.Tensor(X_val))
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Full Loader (for final inference/scoring on all data)
    full_dataset = TensorDataset(torch.Tensor(X_sequences))
    full_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    
    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMAutoencoder(seq_len=SEQ_LEN, n_features=len(features)).to(device)
    
    # Train with Early Stopping
    model = train_model(model, train_loader, val_loader, epochs=20, device=device, patience=3)
    
    # Save Model state dict
    model_path = os.path.join(BASE_DIR, 'models', 'lstm_autoencoder.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Get Anomaly Scores (Reconstruction Error) on FULL dataset
    print("Calculating reconstruction errors...")
    errors = get_reconstruction_error(model, full_loader, device=device)
    
    # Pad the beginning of the dataframe to match sequence length shift
    # Since sequences are [0:30], [1:31]... 
    # The anomaly score corresponds to the window. We can align it to the end of window.
    padding = [np.nan] * SEQ_LEN
    df['reconstruction_error'] = np.concatenate([padding, errors])
    
    # Define Anomaly Threshold
    # Using 95th percentile of reconstruction error as threshold
    threshold = np.nanpercentile(df['reconstruction_error'], 95)
    print(f"Anomaly Threshold (MSE 95%): {threshold:.4f}")
    
    df['is_anomaly'] = df['reconstruction_error'] > threshold
    
    return df

def engineer_features(df, window_seconds=30):
    print("Engineering features...")
    df['hr_mean'] = df['hr_cleaned'].rolling(window=window_seconds, min_periods=1).mean()
    df['hr_std'] = df['hr_cleaned'].rolling(window=window_seconds, min_periods=1).std().fillna(0)
    df['spo2_mean'] = df['spo2_cleaned'].rolling(window=window_seconds, min_periods=1).mean()
    df['spo2_std'] = df['spo2_cleaned'].rolling(window=window_seconds, min_periods=1).std().fillna(0)
    df['hr_trend'] = df['hr_mean'].diff().fillna(0)
    df['spo2_trend'] = df['spo2_mean'].diff().fillna(0)
    df['shock_index'] = df['hr_cleaned'] / df['sbp']
    return df

def calculate_risk_score(df):
    print("Calculating Clinical Risk Scores...")
    # HR Score
    conditions_hr = [
        (df['hr_cleaned'] > 120) | (df['hr_cleaned'] < 40),
        (df['hr_cleaned'] > 100) | (df['hr_cleaned'] < 50),
        (df['hr_trend'] > 0.5)
    ]
    choices_hr = [30, 10, 5]
    df['hr_risk_subscore'] = np.select(conditions_hr, choices_hr, default=0)
    
    # SpO2 Score
    conditions_spo2 = [
        (df['spo2_cleaned'] < 90),
        (df['spo2_cleaned'] < 95),
        (df['spo2_trend'] < -0.2)
    ]
    choices_spo2 = [40, 15, 10]
    df['spo2_risk_subscore'] = np.select(conditions_spo2, choices_spo2, default=0)
    
    # Shock Index
    df['si_risk_subscore'] = np.where(df['shock_index'] > 0.9, 20, 0)
    
    # Total
    df['risk_score'] = df['hr_risk_subscore'] + df['spo2_risk_subscore'] + df['si_risk_subscore']
    df['risk_score'] = df['risk_score'].clip(upper=100)
    
    # Confidence
    vib_max = df['vibration'].quantile(0.99)
    df['confidence_score'] = 1.0 - (df['vibration'] / vib_max)
    df['confidence_score'] = df['confidence_score'].clip(0, 1)
    
    # Alerts
    # Alerts logic with explanation
    risk_threshold = 40
    conf_threshold = 0.5
    window_persistence = 5
    
    # Initial trigger check
    df['high_risk'] = df['risk_score'] > risk_threshold
    df['reliable_data'] = df['confidence_score'] > conf_threshold
    
    # Determine Status
    conditions = [
        (~df['high_risk']), # Normal
        (df['high_risk'] & ~df['reliable_data']), # Suppressed due to low confidence (vibration)
        (df['high_risk'] & df['reliable_data']) # High Risk & Reliable -> Potential Alert
    ]
    choices = ['NORMAL', 'SUPPRESSED (Low Confidence)', 'WARNING (Transient)']
    df['triage_status'] = np.select(conditions, choices, default='UNKNOWN')
    
    # Check Persistence for Critical Alerts
    # We only want to escalate 'WARNING' to 'CRITICAL' if it persists
    # Create a boolean series for 'WARNING' state
    is_warning = df['triage_status'] == 'WARNING (Transient)'
    # Check if the warning state persists for window_persistence
    is_persistent = is_warning.rolling(window=window_persistence).sum() == window_persistence
    
    # Update status to CRITICAL for persistent warnings
    df.loc[is_persistent, 'triage_status'] = 'CRITICAL ALERT'
    
    # Backward fill the Critical status for the duration of the window to capture the onset? 
    # Or just mark the moment it becomes critical? Let's keep it simple: mark the *sequence*.
    # Actually, usually you want to alert *at* the moment it becomes persistent. 
    # But for analysis, seeing the whole block as critical is nice. 
    # Let's stick to the instant trigger logic but maybe mark the whole event in a log.
    
    df['alert_triggered'] = df['triage_status'] == 'CRITICAL ALERT'
    
    # Print Summary of Logic
    print("\n--- Triage Logic Summary ---")
    print(f"Risk Threshold: > {risk_threshold} (Clinical Score)")
    print(f"Confidence Threshold: > {conf_threshold} (1 - Vibration Impact)")
    print(f"Persistence Window: {window_persistence} seconds")
    print("-" * 30)
    print(df['triage_status'].value_counts())
    print("-" * 30)
    
    return df

def plot_analysis(df):
    print("Generating Analysis Plots (Deep Learning Version)...")
    fig, axes = plt.subplots(4, 1, figsize=(15, 16), sharex=True)
    x = df.index
    
    # 1. Vitals & Deep Learning Anomalies
    axes[0].plot(x, df['hr_cleaned'], color='magenta', label='HR', alpha=0.6)
    axes[0].plot(x, df['spo2_cleaned'], color='green', label='SpO2', alpha=0.6)
    # Highlight anomalies
    anomalies = df[df['is_anomaly']]
    if not anomalies.empty:
        axes[0].scatter(anomalies.index, anomalies['hr_cleaned'], color='black', s=10, label='DL Anomaly (High Recons. Error)', zorder=5)
    axes[0].set_title('Task 2A: Deep Learning Anomaly Detection (LSTM Autoencoder)')
    axes[0].set_ylabel('Vitals')
    axes[0].legend(loc='upper right')
    axes[0].grid(alpha=0.3)
    
    # 2. Reconstruction Error (The Metric)
    axes[1].plot(x, df['reconstruction_error'], color='brown', label='Reconstruction Error (MSE)')
    threshold = np.nanpercentile(df['reconstruction_error'], 95)
    axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[1].set_title('Model Performance: Reconstruction Error')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(alpha=0.3)
    
    # 3. Risk Score
    axes[2].plot(x, df['risk_score'], color='orange', label='Clinical Risk Score')
    axes[2].axhline(y=40, color='red', linestyle='--', label='Alert Threshold')
    axes[2].set_title('Task 2B: Clinical Risk Scoring')
    axes[2].set_ylabel('Score')
    axes[2].legend(loc='upper right')
    axes[2].grid(alpha=0.3)
    
    # 4. Alerts
    axes[3].plot(x, df['confidence_score'], color='blue', label='Confidence')
    
    # Critical Alerts
    critical = df[df['triage_status'] == 'CRITICAL ALERT']
    if not critical.empty:
        axes[3].scatter(critical.index, [1.05]*len(critical), color='red', marker='v', s=30, label='CRITICAL ALERT (Persistent)', zorder=5)
        
    # Suppressed Alerts
    suppressed = df[df['triage_status'] == 'SUPPRESSED (Low Confidence)']
    if not suppressed.empty:
        axes[3].scatter(suppressed.index, [1.05]*len(suppressed), color='gray', marker='x', s=20, label='SUPPRESSED (High Vib)', zorder=4)

    axes[3].axhline(y=0.5, color='blue', linestyle=':', label='Confidence Threshold')
    axes[3].set_title('Triage Decisions: Confidence & Alerts')
    axes[3].set_ylim(0, 1.2)
    axes[3].legend(loc='upper right', fontsize='small')
    axes[3].grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, 'docs', 'anomaly_risk_analysis_dl.png')
    plt.savefig(output_path)
    print(f"Analysis plot saved to {output_path}")

def main():
    df = load_data(INPUT_FILE)
    if df is not None:
        df = engineer_features(df)
        df = run_deep_anomaly_detection(df)
        df = calculate_risk_score(df)
        plot_analysis(df)
        
        outfile = os.path.join(BASE_DIR, 'data', 'smart_ambulance_risk_scored_dl.csv')
        df.to_csv(outfile, index=False)
        print(f"Results saved to {outfile}")

if __name__ == "__main__":
    main()
