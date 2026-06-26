# 🛰️ NASA Telemetry Anomaly Detection API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.138+-green.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-red.svg)](https://pytorch.org/)

**Real-time anomaly detection for multivariate sensor telemetry** – built for the NASA SMAP & MSL spacecraft datasets. This system ingests streaming sensor readings and returns anomaly scores within **<50ms per sample** on CPU, making it suitable for industrial IoT and aerospace monitoring applications.

---

## 📌 Table of Contents
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Performance Benchmark](#performance-benchmark)
- [How It Works (Q1–Q7)](#how-it-works-q1q7)
- [Tech Stack](#tech-stack)
- [Author](#author)

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Multivariate LSTM Autoencoder** | Captures complex inter-sensor dependencies across 82 telemetry channels simultaneously. |
| **99th Percentile Thresholding** | Statistically robust anomaly scoring (non-parametric, handles skewed error distributions). |
| **Adaptive Thresholding (Q4)** | Dynamically adjusts to machine warm‑up periods, reducing false positives by 30%+. |
| **50ms Real-Time Inference (Q5)** | Optimized with TorchScript, `torch.inference_mode()`, and single-threaded CPU execution. |
| **Online Learning & Drift Detection (Q6)** | Tracks concept drift in real-time using Welford's algorithm – no retraining required. |
| **Production-Ready API (Q7)** | FastAPI service with `/stream` (ingestion) and `/health` (monitoring) endpoints. |
| **CPU-Only** | Runs entirely on standard CPUs – no GPU required (hard constraint met). |

---

## 🧠 Architecture Overview

```mermaid
graph LR
    A[Sensor Telemetry<br>82 Channels] --> B[Preprocessing<br>Missing Value Imputation]
    B --> C[Sliding Window<br>50 Timesteps]
    C --> D[LSTM Autoencoder<br>Encoder + Decoder]
    D --> E[Reconstruction Error<br>MSE]
    E --> F[Adaptive Threshold<br>95th %ile of recent errors]
    F --> G[Anomaly Decision<br>Pass / Fail]
    E --> H[Drift Detector<br>Welford's Running Mean]
    H --> I[/health Endpoint<br>Drift Status]
