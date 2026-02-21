import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Load Data
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_PATH = os.path.join(BASE_DIR, "data", "smart_ambulance_risk_scored_dl.csv")
print(f"Loading data from {FILE_PATH}...")
df = pd.read_csv(FILE_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# --- 1. Define Ground Truth & Predictions ---
# We define a 'True Critical Event' as when the patient is labeled 'critical'.
# We define a 'System Alert' as when 'triage_status' is 'CRITICAL ALERT'.

df['ground_truth_critical'] = df['patient_type'] == 'critical'
df['system_alert_critical'] = df['triage_status'] == 'CRITICAL ALERT'

y_true = df['ground_truth_critical']
y_pred = df['system_alert_critical']

# --- 2. Calculate Metrics ---
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# False Alert Rate (FAR): False Positives / Total Non-Critical Samples
# Or False Alerts per Hour. Let's do both.
far_ratio = fp / (tn + fp) 
total_duration_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
false_alerts_per_hour = fp / total_duration_hours if total_duration_hours > 0 else 0

print("\n" + "="*40)
print("TASK 3A: METRICS DEFINITION & REPORT")
print("="*40)
print(f"Total Samples: {len(df)}")
print(f"True Critical Samples: {y_true.sum()}")
print(f"System Alerts Triggered: {y_pred.sum()}")
print("-" * 30)
print(f"Precision: {precision:.4f} (Probability that an alert is real)")
print(f"Recall:    {recall:.4f}    (Probability that a critical event is detected)")
print(f"F1 Score:  {f1:.4f}")
print("-" * 30)
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"False Alert Rate (Sample-wise): {far_ratio:.4f}")
print(f"False Alerts Per Sample Hour:   {false_alerts_per_hour:.2f}")

# --- 3. Alert Latency ---
# Calculate time from 'critical' onset to first 'CRITICAL ALERT'
print("\nCalculating Alert Latency...")
latencies = []

# Group by patient (assuming consecutive rows per patient are time-ordered)
for pid, group in df.groupby('patient_id'):
    # Find transitions to critical
    group = group.sort_values('timestamp')
    group['is_critical_start'] = (group['patient_type'] == 'critical') & (group['patient_type'].shift(1) != 'critical')
    critical_starts = group[group['is_critical_start']].index
    
    for start_idx in critical_starts:
        start_time = group.loc[start_idx, 'timestamp']
        # Find first alert AFTER this start time
        # We look at the slice from start_time onwards
        future_mask = (group['timestamp'] >= start_time) & (group['triage_status'] == 'CRITICAL ALERT')
        future_alerts = group.loc[future_mask]
        
        if not future_alerts.empty:
            first_alert_time = future_alerts.iloc[0]['timestamp']
            latency = (first_alert_time - start_time).total_seconds()
            latencies.append(latency)

avg_latency = np.mean(latencies) if latencies else 0
print(f"Average Alert Latency: {avg_latency:.2f} seconds")

# --- 4. Acceptable Errors Discussion ---
print("\n" + "-"*40)
print("Contextual Interpretation:")
print("1. False Positives (FP): annoying but acceptable. Better to pull over for a Check Engine light that is a sensor glitch than to ignore smoke.")
print("2. False Negatives (FN): UNACCEPTABLE. Missing a cardiac arrest is fatal.")
print("   -> Current System Design prioritizes High Validity (Logic) but might have lower Recall due to the 5s persistence window.")
print("-"*40)


# --- 5. Failure Analysis (Task 3B) ---
print("\n" + "="*40)
print("TASK 3B: FAILURE ANALYSIS")
print("="*40)

# Case 1: False Negative (Critical but No Alert)
# Find a 'critical' row where system_alert_critical is False
fn_indices = df[(df['patient_type'] == 'critical') & (df['triage_status'] != 'CRITICAL ALERT')].index
if not fn_indices.empty:
    idx = fn_indices[0]
    row = df.loc[idx]
    print("\n[FAILURE CASE 1: FALSE NEGATIVE]")
    print(f"Timestamp: {row['timestamp']}")
    print(f"Patient Type: {row['patient_type']}")
    print(f"Triage Status: {row['triage_status']}")
    print(f"Risk Score: {row['risk_score']} (Threshold: 40)")
    print(f"Confidence Score: {row['confidence_score']:.2f} (Threshold: 0.5)")
    print(f"Vibration: {row['vibration']}")
    print("ANALYSIS: Use the values above to explain why it failed (e.g., suppressed by vibration? Risk score too low?)")

# Case 2: False Positive (Normal/Stable but Alert)
# Find a generic non-critical row where alert is True
fp_indices = df[(df['patient_type'].isin(['normal', 'stable'])) & (df['triage_status'] == 'CRITICAL ALERT')].index
if not fp_indices.empty:
    idx = fp_indices[0]
    row = df.loc[idx]
    print("\n[FAILURE CASE 2: FALSE POSITIVE]")
    print(f"Timestamp: {row['timestamp']}")
    print(f"Patient Type: {row['patient_type']}")
    print(f"Triage Status: {row['triage_status']}")
    print(f"Risk Score: {row['risk_score']}")
    print(f"Shock Index: {row['shock_index']:.2f}")
    print("ANALYSIS: Why did it trigger? (e.g., specific sensor spike?)")

# Case 3: Late Detection (High Latency)
# We can just look at a latency > 10s if we calculated them
# Or look for 'SUPPRESSED' state while Critical
suppressed_critical = df[(df['patient_type'] == 'critical') & (df['triage_status'] == 'SUPPRESSED (Low Confidence)')]
if not suppressed_critical.empty:
    idx = suppressed_critical.index[0]
    row = df.loc[idx]
    print("\n[FAILURE CASE 3: SUPPRESSED CRITICAL EVENT]")
    print(f"Timestamp: {row['timestamp']}")
    print(f"Patient Type: {row['patient_type']}")
    print(f"Triage Status: {row['triage_status']}")
    print(f"Vibration: {row['vibration']}")
    print("ANALYSIS: The patient was critical, but the ambulance vibration caused the system to suppress the alert.")
