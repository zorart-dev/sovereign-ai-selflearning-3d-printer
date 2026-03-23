"""Perception encoder – fuses vision and sensor data."""

import torch
import torch.nn as nn


class TinyVisionCNN(nn.Module):
    """Lightweight CNN for camera frames."""
    
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_dim)
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))


class Perception(nn.Module):
    """Fuse sensor and vision data."""
    
    def __init__(self, sensor_dim: int = 8, vision_dim: int = 64, fused_dim: int = 128):
        super().__init__()
        self.vision_encoder = TinyVisionCNN(out_dim=vision_dim)
        self.sensor_fc = nn.Linear(sensor_dim, fused_dim // 2)
        self.fusion = nn.Linear(vision_dim + fused_dim // 2, fused_dim)
    
    def forward(self, sensors: torch.Tensor, vision: torch.Tensor = None):
        s = self.sensor_fc(sensors)
        if vision is not None:
            v = self.vision_encoder(vision)
            fused = torch.cat([s, v], dim=1)
        else:
            fused = torch.cat([s, torch.zeros_like(s)], dim=1)
        return self.fusion(fused)