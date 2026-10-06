"""
generate_data.py — Synthetic Healthcare Dataset Generator for PrivaQuery
Generates 5,000 rows of realistic patient data and saves to healthcare_data.csv
"""

import uuid
import numpy as np
import pandas as pd

# Reproducible randomness
rng = np.random.default_rng(seed=42)

NUM_ROWS = 5_000

data = pd.DataFrame({
    "PatientID": [str(uuid.uuid4()) for _ in range(NUM_ROWS)],
    "Age": rng.integers(low=18, high=91, size=NUM_ROWS),                 # 18–90
    "Income": rng.integers(low=30_000, high=150_001, size=NUM_ROWS),     # 30k–150k
    "Cholesterol": rng.integers(low=120, high=301, size=NUM_ROWS),       # 120–300
    "BloodPressure": rng.integers(low=80, high=181, size=NUM_ROWS),      # 80–180
    "HeartDisease": rng.integers(low=0, high=2, size=NUM_ROWS),          # 0 or 1
})

data.to_csv("healthcare_data.csv", index=False)
print(f"[OK] healthcare_data.csv generated with {NUM_ROWS} rows.")
print(data.head())
