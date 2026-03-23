# 🔥 Sovereign-AI: Autonomous 3D Printer Optimization System

![Version](https://img.shields.io/badge/version-5.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)

**Sovereign-AI** is a production-grade, edge-deployed autonomous learning system that uses Proximal Policy Optimization (PPO) with LSTM memory to continuously optimize 3D printer settings. Designed for Raspberry Pi 4, it learns from camera feedback and hardware sensors to improve print quality without human intervention.

## 🎯 Key Features

- **Full PPO Implementation** – Clipped objectives, GAE, entropy regularization
- **LSTM Memory** – Temporal reasoning for printer dynamics
- **Vision Fusion** – CNN encodes camera frames, fused with sensor data
- **Adaptive Safety** – Hardware-aware constraints that learn from failures
- **Edge Optimized** – <220MB RAM, CPU-only inference, ~30fps
- **Persistent Learning** – Survives power cycles, accumulates experience
- **Multi-Version Progression** – v1 (basic active learning) → v5 (full LSTM-PPO)
- **Federated Ready** – Multi-printer learning coordination (stub included)
- **Production Hardening** – Error handling, rate limiting, emergency stop

## 📊 Architecture
