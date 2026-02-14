
# Documentation: Assumptions, Signal Meanings, and Limitations

## 1. Assumptions
- **Baseline Vitals**: Patient baselines are derived from statistical averages (mean/std) of the `human_vital_signs_dataset_2024.csv`. We assume this dataset is representative of the general population.
- **Patient Types**: Patients are categorized into four types: 'normal', 'stable', 'deteriorating', and 'critical'. Each type has defined ranges for vitals and stress levels.
- **Motion Simulation**: Motion is simulated based on three scenarios: 'urban', 'highway', and 'rural'. We assume these scenarios cover the typical transport conditions for an ambulance.
- **Stress Influence**: We assume that vibration and patient condition directly influence 'stress', which in turn affects heart rate, blood pressure, and SpO2.
- **Linear Deterioration**: For 'deteriorating' and 'critical' patients, health metrics are assumed to deteriorate linearly over time (`deterioration_rate * i`).

## 2. Signal Meanings
- **Heart Rate (HR)**: Beats per minute. Influenced by baseline condition, stress, and deterioration.
- **SpO2**: Oxygen saturation percentage. Influenced by baseline, stress, deterioration, and motion artifacts (vibration > 1.2 triggers drops).
- **Systolic/Diastolic Blood Pressure (SBP/DBP)**: mmHg. SBP is correlated with HR and stress. DBP is derived from SBP and stress.
- **Vibration (Motion)**: Calculated magnitude of 3-axis acceleration (x, y, z). Represents the physical movement of the ambulance and patient.
  - **x, y, z**: Acceleration components.
  - **vib**: Magnitude of vibration ($\sqrt{x^2 + y^2 + z^2}$).
- **Stress**: A derived internal metric (0.0 - 1.0) representing the physiological stress load on the patient.

## 3. Limitations
- **Simplified Physiology**: The relationships between vitals (e.g., HR increasing SBP) are modeled using simple linear equations and may not capture complex non-linear physiological responses.
- **Synthetic Noise**: Road noise and motion artifacts are generated using random distributions (Gaussian/Uniform) and may not perfectly reflect real-time sensor noise characteristics.
- **Discrete Scenarios**: Journey routes are fixed to single scenarios ('urban' OR 'highway') for the entire duration, whereas real journeys often involve a mix.
- **Univariate Deterioration**: Deterioration is modeled as a constant rate, whereas real medical emergencies can have sudden, non-linear crash events.
- **Duration**: Journeys are simulated for short durations (30-60 mins), which may not cover long-distance transfer dynamics.
