import requests
import json
import time

# API Endpoint
URL = "http://127.0.0.1:8000/predict"

def test_api(scenario_name, data):
    print(f"\nTesting Scenario: {scenario_name}")
    try:
        response = requests.post(URL, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"Status Code: {response.status_code}")
            print("Response:")
            print(json.dumps(result, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Request Failed: {e}")

if __name__ == "__main__":
    print("Sending requests to Smart Ambulance API...")
    
    # Scenario 1: Normal Patient
    normal_data = {
        "timestamp": "2026-02-14T12:00:00",
        "heart_rate": 75.0,
        "spo2": 98.0,
        "sbp": 120.0,
        "dbp": 80.0,
        "vibration": 0.1
    }
    test_api("Normal Patient", normal_data)
    
    # Scenario 2: High Risk but High Vibration (Suppressed?)
    # Note: API might need history to trend, so we send a few requests to fill buffer
    print("\nSimulating stream for Warning...")
    warning_data = {
        "timestamp": "2026-02-14T12:00:01",
        "heart_rate": 130.0, # High
        "spo2": 88.0,        # Low
        "sbp": 90.0,
        "dbp": 60.0,
        "vibration": 0.8     # High Vibration -> Should lower confidence
    }
    
    # Send a few times to simulate persistence/buffering if needed
    for i in range(3):
        test_api(f"High Risk / High Vib (Frame {i+1})", warning_data)
        time.sleep(0.1)

    # Scenario 3: Critical (Low Vibration)
    critical_data = {
        "timestamp": "2026-02-14T12:00:05",
        "heart_rate": 140.0,
        "spo2": 85.0,
        "sbp": 80.0,
        "dbp": 50.0,
        "vibration": 0.0     # Low Vibration -> High Confidence
    }
    test_api("Critical Event", critical_data)
