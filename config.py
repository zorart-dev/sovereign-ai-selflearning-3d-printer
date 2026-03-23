"""
Sovereign-v5.0 Configuration Management
Production-grade config with validation and sensible defaults
"""

from dataclasses import dataclass, field
from typing import List, Optional
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LearningConfig:
    """Learning hyperparameters"""
    lr: float = 3e-4
    lr_perception: float = 1.5e-4  # Slower for perception
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    batch_size: int = 64

    def validate(self):
        """Validate config values"""
        assert 0 < self.lr < 1.0, f"Invalid lr: {self.lr}"
        assert 0 < self.gamma < 1.0, f"Invalid gamma: {self.gamma}"
        assert 0 < self.gae_lambda < 1.0, f"Invalid gae_lambda: {self.gae_lambda}"
        assert 0 < self.clip_ratio < 1.0, f"Invalid clip_ratio: {self.clip_ratio}"
        assert self.batch_size > 0, f"Invalid batch_size: {self.batch_size}"
        assert self.epochs_per_update > 0, f"Invalid epochs_per_update: {self.epochs_per_update}"


@dataclass
class HardwareConfig:
    """Hardware-specific constraints and limits"""
    nozzle_min: float = 180.0
    nozzle_max: float = 260.0
    speed_min: float = 20.0
    speed_max: float = 180.0
    bed_min: float = 30.0
    bed_max: float = 120.0
    safe_margin: float = 8.0
    control_interval: float = 0.8  # seconds between actions
    camera_id: int = 0

    def validate(self):
        """Validate hardware constraints"""
        assert self.nozzle_min < self.nozzle_max, "Invalid nozzle temperature range"
        assert self.speed_min < self.speed_max, "Invalid speed range"
        assert self.bed_min < self.bed_max, "Invalid bed temperature range"
        assert self.control_interval > 0, "Invalid control interval"


@dataclass
class CurriculumConfig:
    """Curriculum learning parameters"""
    stages: int = 3
    reward_scaling: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    stage_duration: int = 1000  # steps between curriculum changes

    def validate(self):
        """Validate curriculum config"""
        assert self.stages > 0, "Invalid number of stages"
        assert len(self.reward_scaling) == self.stages, "Reward scaling must match number of stages"
        assert all(s > 0 for s in self.reward_scaling), "All reward scales must be positive"


@dataclass
class FederatedConfig:
    """Federated learning parameters"""
    enabled: bool = False
    server_url: Optional[str] = None
    sync_interval: int = 100  # steps between syncs
    node_id: Optional[str] = None

    def validate(self):
        """Validate federated config"""
        if self.enabled:
            assert self.server_url is not None, "Server URL required for federated learning"


@dataclass
class Config:
    """Master configuration"""
    learning: LearningConfig = field(default_factory=LearningConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)

    # Paths
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    log_level: str = "INFO"

    # Device
    device: str = "auto"  # 'cpu', 'cuda', or 'auto'

    def __post_init__(self):
        """Initialize and validate"""
        self.validate()
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def validate(self):
        """Validate all sub-configs"""
        self.learning.validate()
        self.hardware.validate()
        self.curriculum.validate()
        self.federated.validate()
        logger.info("Configuration validated successfully")

    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'learning': self.learning.__dict__,
            'hardware': self.hardware.__dict__,
            'curriculum': self.curriculum.__dict__,
            'federated': self.federated.__dict__,
            'checkpoint_dir': self.checkpoint_dir,
            'log_dir': self.log_dir,
        }

    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")

    @staticmethod
    def load(path: str) -> 'Config':
        """Load config from JSON"""
        with open(path, 'r') as f:
            data = json.load(f)
        config = Config()
        # Update fields from loaded data
        for key, value in data.items():
            if hasattr(config, key) and isinstance(value, dict):
                for k, v in value.items():
                    setattr(getattr(config, key), k, v)
        config.validate()
        logger.info(f"Config loaded from {path}")
        return config


# Default global config instance
DEFAULT_CONFIG = Config()
