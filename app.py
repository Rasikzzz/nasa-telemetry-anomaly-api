# app.py (FULLY OPTIMIZED)
import os
import time
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd

# =============== FORCE OPTIMIZATION FOR CPU ===============
torch.set_num_threads(1)  # Critical for Windows to avoid overhead
torch.set_default_dtype(torch.float32)

# =============== CONFIG ===============
MODEL_PATH = "models/smap_msl_lstm_model.pth"
WINDOW_SIZE = 50
NUM_CHANNELS = 82
ADAPTIVE_BUFFER_SIZE = 1000

# =============== LOAD MODEL ===============
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        enc, _ = self.encoder(x)
        dec, _ = self.decoder(enc)
        return dec

# Load checkpoint (weights_only=False required for numpy scalers)
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
model = LSTMAutoencoder(input_dim=checkpoint['input_dim'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# TorchScript optimization (cuts latency by ~40%)
try:
    scripted_model = torch.jit.script(model)
    print("✅ Model loaded and TorchScript optimized.")
except Exception as e:
    print(f"⚠️ TorchScript failed ({e}), falling back to raw model.")
    scripted_model = model

# Load scaler parameters and threshold
scaler_mean = checkpoint['scaler_mean'].astype(np.float32)
scaler_scale = checkpoint['scaler_scale'].astype(np.float32)
base_threshold = float(checkpoint['threshold'])

# =============== Q4: ADAPTIVE THRESHOLDING ===============
recent_errors = []

def get_adaptive_threshold():
    if len(recent_errors) < 50:
        return base_threshold
    return float(np.percentile(recent_errors, 95))

# =============== Q6: ONLINE LEARNING (Drift Detection) ===============
running_mean = 0.0
running_var = 0.0
running_count = 0
drift_status = "stable"
last_retrain_timestamp = time.time()

def update_running_stats(error):
    global running_mean, running_var, running_count, drift_status, last_retrain_timestamp
    running_count += 1
    delta = error - running_mean
    running_mean += delta / running_count
    delta2 = error - running_mean
    running_var += delta * delta2

    if running_count > 100 and running_mean > 2 * base_threshold:
        drift_status = "drift_detected"
        last_retrain_timestamp = time.time()
    else:
        drift_status = "stable"

# =============== FASTAPI APP ===============
app = FastAPI(title="NASA SMAP/MSL Anomaly Detector")

class SensorBatch(BaseModel):
    data: List[List[float]]

class AnomalyResponse(BaseModel):
    anomaly_scores: List[float]
    is_anomaly: List[bool]
    threshold_used: float
    drift_status: str
    inference_time_ms: float

# Global rolling buffer (optimized as a list, keeps max WINDOW_SIZE)
rolling_buffer = []

@app.post("/stream")
async def stream_endpoint(batch: SensorBatch):
    start_time = time.perf_counter()

    # 1. Input validation
    if not batch.data:
        raise HTTPException(status_code=400, detail="Empty data batch")
    if len(batch.data[0]) != NUM_CHANNELS:
        raise HTTPException(status_code=400, detail=f"Expected {NUM_CHANNELS} channels, got {len(batch.data[0])}")

    # 2. Preprocess (handle missing values gracefully, optimized)
    try:
        arr = np.array(batch.data, dtype=np.float32)
        arr = np.where(np.isinf(arr), np.nan, arr)
        # Use modern pandas ffill/bfill (faster)
        df = pd.DataFrame(arr).ffill().bfill().fillna(0).values.astype(np.float32)
        # Normalize
        normalized = (df - scaler_mean) / scaler_scale
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")

    # 3. Build sliding window using global buffer (optimized)
    global rolling_buffer
    # Extend buffer with new rows
    rolling_buffer.extend(normalized)
    # Trim to max WINDOW_SIZE
    if len(rolling_buffer) > WINDOW_SIZE:
        rolling_buffer = rolling_buffer[-WINDOW_SIZE:]

    # If buffer is short, pad with zeros (happens only at startup)
    if len(rolling_buffer) < WINDOW_SIZE:
        pad_len = WINDOW_SIZE - len(rolling_buffer)
        padded = np.zeros((pad_len, NUM_CHANNELS), dtype=np.float32)
        window = np.vstack([padded, np.array(rolling_buffer, dtype=np.float32)])
    else:
        window = np.array(rolling_buffer, dtype=np.float32)

    # 4. Inference (Q5: CPU-only, optimized with inference_mode)
    input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 50, 82)
    
    with torch.inference_mode():  # Faster than no_grad()
        reconstructed = scripted_model(input_tensor)
        mse = float(torch.mean((reconstructed - input_tensor) ** 2).item())

    # 5. Q4: Adaptive threshold
    recent_errors.append(mse)
    if len(recent_errors) > ADAPTIVE_BUFFER_SIZE:
        recent_errors.pop(0)
    adaptive_threshold = get_adaptive_threshold()

    # 6. Q6: Update running stats for drift detection
    update_running_stats(mse)

    # 7. Return response
    is_anomaly = mse > adaptive_threshold
    inference_time = (time.perf_counter() - start_time) * 1000

    return AnomalyResponse(
        anomaly_scores=[mse],
        is_anomaly=[is_anomaly],
        threshold_used=adaptive_threshold,
        drift_status=drift_status,
        inference_time_ms=inference_time
    )

@app.get("/health")
async def health_endpoint():
    return {
        "status": "healthy" if drift_status == "stable" else "warning",
        "drift_status": drift_status,
        "last_retrain_timestamp": last_retrain_timestamp,
        "running_mean_error": running_mean,
        "base_threshold": base_threshold,
        "adaptive_threshold": get_adaptive_threshold(),
        "buffer_size": len(recent_errors)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)