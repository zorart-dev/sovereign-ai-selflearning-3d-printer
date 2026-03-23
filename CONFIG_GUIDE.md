# Configuration Reference Guide: Sovereign-v5.0

Complete reference for all configuration parameters with explanations, examples, and tuning recommendations.

## Quick Reference

### Minimal Configuration (Testing)
```json
{
  "learning": {"learning_rate": 3e-4},
  "hardware": {"camera_id": 0},
  "federated": {"enabled": false}
}
```

### Production Configuration
```json
{
  "learning": {
    "learning_rate": 1e-4,
    "entropy_coef": 0.001,
    "value_coef": 0.5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5
  },
  "hardware": {
    "nozzle_min": 180,
    "nozzle_max": 260,
    "speed_min": 20,
    "speed_max": 180,
    "safe_margin": 5.0,
    "camera_id": 0
  }
}
```

---

## Configuration Sections

### 1. LEARNING CONFIGURATION

#### `learning_rate`
- **Type**: float
- **Default**: 3e-4
- **Range**: 1e-5 to 1e-2
- **Description**: Learning rate for policy optimization
- **Effects**:
  - Too high (>1e-2): Divergence, NaN losses
  - Too low (<1e-5): Extremely slow convergence
  - Recommended: 1e-4 to 1e-3
- **Tuning Tips**:
  - Start with 1e-4, adjust based on loss behavior
  - Use learning rate schedule for longer training
  - Reduce if training becomes unstable

**Examples**:
```python
# Fast learning (exploration phase)
config.learning.learning_rate = 5e-3

# Careful learning (fine-tuning)
config.learning.learning_rate = 1e-5

# Balanced (default)
config.learning.learning_rate = 3e-4
```

---

#### `entropy_coef`
- **Type**: float
- **Default**: 0.01
- **Range**: 0 to 0.1
- **Description**: Entropy regularization coefficient
- **Effects**:
  - Higher: More exploration, less deterministic
  - Lower: More exploitation, more greedy
- **Tuning Tips**:
  - Increase if policy becomes too deterministic
  - Decrease if too much random exploration
  - 0.001-0.01 works for most cases

**Examples**:
```python
# High exploration
config.learning.entropy_coef = 0.05

# Low exploration
config.learning.entropy_coef = 0.001

# Balanced (default)
config.learning.entropy_coef = 0.01
```

---

#### `value_coef`
- **Type**: float
- **Default**: 0.5
- **Range**: 0.1 to 1.0
- **Description**: Weight for critic (value) loss
- **Effects**:
  - Higher: Better value estimation
  - Lower: Policy-focused learning
- **Tuning Tips**:
  - Use 0.5 as starting point
  - Increase if value loss is high
  - Decrease if policy loss plateaus

**Examples**:
```python
# Value-focused
config.learning.value_coef = 0.8

# Policy-focused
config.learning.value_coef = 0.2

# Balanced (default)
config.learning.value_coef = 0.5
```

---

#### `gamma`
- **Type**: float
- **Default**: 0.99
- **Range**: 0.9 to 0.999
- **Description**: Discount factor for future rewards
- **Effects**:
  - High (0.99+): Consider far future
  - Low (0.9): Focus on immediate rewards
- **Tuning Tips**:
  - Use 0.99-0.999 for most tasks
  - Lower for time-sensitive tasks
  - Don't change without good reason

**Examples**:
```python
# Long-horizon rewards
config.learning.gamma = 0.995

# Short-horizon rewards
config.learning.gamma = 0.95

# Standard (default)
config.learning.gamma = 0.99
```

---

#### `gae_lambda`
- **Type**: float
- **Default**: 0.95
- **Range**: 0.8 to 0.99
- **Description**: GAE (Generalized Advantage Estimation) lambda
- **Effects**:
  - Higher (0.95+): More variance, less bias
  - Lower (0.8): Less variance, more bias
- **Tuning Tips**:
  - Use 0.95-0.98 for best bias-variance trade-off
  - Higher for exploration phases
  - Lower for fine-tuning

**Examples**:
```python
# High variance (exploration)
config.learning.gae_lambda = 0.98

# Low variance (stability)
config.learning.gae_lambda = 0.90

# Balanced (default)
config.learning.gae_lambda = 0.95
```

---

#### `max_grad_norm`
- **Type**: float
- **Default**: 0.5
- **Range**: 0.1 to 1.0
- **Description**: Maximum gradient norm for clipping
- **Effects**:
  - Prevents gradient explosions
  - Stabilizes training
- **Tuning Tips**:
  - Use 0.5 for most cases
  - Increase if gradient clipping too aggressive
  - Decrease if gradients explode

**Examples**:
```python
# Conservative clipping
config.learning.max_grad_norm = 0.3

# Liberal clipping
config.learning.max_grad_norm = 1.0

# Balanced (default)
config.learning.max_grad_norm = 0.5
```

---

### 2. HARDWARE CONFIGURATION

#### `nozzle_min` / `nozzle_max`
- **Type**: float (temperature in °C)
- **Default**: 180 / 260
- **Range**: 150-300°C (reasonable: 170-280°C)
- **Description**: Hard constraints on nozzle temperature
- **Effects**:
  - Prevents hardware damage
  - Limits exploration space
- **Tuning Tips**:
  - Set based on filament type and hardware
  - PLA: 190-220°C
  - ABS: 210-250°C
  - PETG: 220-250°C

**Filament-Specific Examples**:
```python
# PLA
config.hardware.nozzle_min = 190
config.hardware.nozzle_max = 220

# ABS
config.hardware.nozzle_min = 210
config.hardware.nozzle_max = 250

# PETG
config.hardware.nozzle_min = 220
config.hardware.nozzle_max = 250

# Default
config.hardware.nozzle_min = 180
config.hardware.nozzle_max = 260
```

---

#### `speed_min` / `speed_max`
- **Type**: float (speed in mm/min)
- **Default**: 20 / 180
- **Range**: 5-500 mm/min
- **Description**: Hard constraints on extrusion speed
- **Effects**:
  - Too slow: Poor quality, cold extrusion
  - Too fast: Print failures, quality issues
- **Tuning Tips**:
  - Start conservative, expand as training progresses
  - Printer-dependent, test manually first
  - Quality prints typically 30-100 mm/min

**Printer-Specific Examples**:
```python
# Fast printer
config.hardware.speed_min = 40
config.hardware.speed_max = 200

# Slow/high-precision
config.hardware.speed_min = 15
config.hardware.speed_max = 100

# Default
config.hardware.speed_min = 20
config.hardware.speed_max = 180
```

---

#### `safe_margin`
- **Type**: float (temperature offset in °C)
- **Default**: 5.0
- **Range**: 1.0 to 20.0
- **Description**: Margin around unsafe temperature ranges
- **Effects**:
  - Larger: More conservative, less exploration
  - Smaller: More aggressive, higher risk
- **Tuning Tips**:
  - Start with 5°C, adjust based on failures
  - Higher for valuable prints
  - Lower for experimental training

**Examples**:
```python
# Very conservative (valuable prints)
config.hardware.safe_margin = 10.0

# Standard (default)
config.hardware.safe_margin = 5.0

# Aggressive (experimental)
config.hardware.safe_margin = 2.0
```

---

#### `camera_id`
- **Type**: int
- **Default**: 0
- **Range**: 0-9
- **Description**: OpenCV camera device ID
- **Effects**:
  - Selects which camera input to use
  - On Linux: /dev/video{id}
  - On Windows: COM{id}
- **Tuning Tips**:
  - Use 0 for single camera setup
  - Test camera IDs with:
    ```python
    import cv2
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): print(f"Camera {i}: OK")
    ```

**Examples**:
```python
# Single camera (default)
config.hardware.camera_id = 0

# Secondary camera
config.hardware.camera_id = 1

# Specific device
config.hardware.camera_id = 2
```

---

### 3. CURRICULUM CONFIGURATION

#### `curriculum_stages`
- **Type**: list of floats
- **Default**: [0.5, 1.0, 2.0]
- **Description**: Reward scaling factors for each stage
- **Effects**:
  - Controls difficulty progression
  - Matches training progress
- **Tuning Tips**:
  - Start with stage 0 (easy)
  - Progress to stage 1 (normal)
  - End with stage 2 (hard)
  - Can add more stages for longer training

**Examples**:
```python
# Simple curriculum (2 stages)
config.learning.curriculum_stages = [0.5, 1.0]

# Complex curriculum (5 stages)
config.learning.curriculum_stages = [0.25, 0.5, 1.0, 2.0, 4.0]

# Default (3 stages)
config.learning.curriculum_stages = [0.5, 1.0, 2.0]
```

---

#### `curriculum_steps_per_stage`
- **Type**: int
- **Default**: 1000
- **Range**: 100 to 10000
- **Description**: Training steps per curriculum stage
- **Effects**:
  - More steps: More time to master each stage
  - Fewer steps: Faster progression
- **Tuning Tips**:
  - Use 1000 for standard training
  - Reduce to 500 for quick experiments
  - Increase to 2000+ for final refinement

**Examples**:
```python
# Quick iteration
config.learning.curriculum_steps_per_stage = 500

# Standard training
config.learning.curriculum_steps_per_stage = 1000

# Careful training
config.learning.curriculum_steps_per_stage = 2000
```

---

### 4. FEDERATED CONFIGURATION

#### `enabled`
- **Type**: bool
- **Default**: False
- **Description**: Enable federated learning
- **Effects**:
  - True: Participate in distributed training
  - False: Local training only

**Examples**:
```python
# Local training
config.federated.enabled = False

# Distributed training
config.federated.enabled = True
```

---

#### `node_id`
- **Type**: string
- **Default**: "printer_0"
- **Description**: Unique identifier for this node
- **Effects**:
  - Used in federated aggregation
  - Identifies source in server logs

**Examples**:
```python
# Single printer
config.federated.node_id = "printer_1"

# Lab setup
config.federated.node_id = "lab_3_printer_a"

# Cloud deployment
config.federated.node_id = "aws_us_east_1_device_42"
```

---

#### `server_url`
- **Type**: string
- **Default**: None
- **Description**: Central federated server URL
- **Effects**:
  - None: Local mode (no sync)
  - URL: Connect to federated server

**Examples**:
```python
# Local development
config.federated.server_url = None

# Lab server
config.federated.server_url = "http://192.168.1.100:5001"

# Cloud server
config.federated.server_url = "https://federated.sovereign.ai:443"
```

---

#### `sync_frequency`
- **Type**: int
- **Default**: 100
- **Range**: 10 to 1000
- **Description**: Steps between federated syncs
- **Effects**:
  - More frequent: More communication, fresher models
  - Less frequent: Less communication, more local training

**Examples**:
```python
# Frequent sync (high bandwidth)
config.federated.sync_frequency = 50

# Standard sync
config.federated.sync_frequency = 100

# Infrequent sync (low bandwidth)
config.federated.sync_frequency = 500
```

---

## Preset Configurations

### High-Precision Training
```json
{
  "learning": {
    "learning_rate": 1e-4,
    "entropy_coef": 0.001,
    "gamma": 0.995,
    "gae_lambda": 0.98
  },
  "hardware": {
    "nozzle_min": 190,
    "nozzle_max": 235,
    "safe_margin": 3.0
  }
}
```

### Fast Convergence
```json
{
  "learning": {
    "learning_rate": 5e-3,
    "entropy_coef": 0.05,
    "gamma": 0.95,
    "gae_lambda": 0.9
  },
  "hardware": {
    "safe_margin": 10.0
  }
}
```

### Conservative (Safety-First)
```json
{
  "learning": {
    "learning_rate": 1e-5,
    "entropy_coef": 0.001,
    "max_grad_norm": 0.3
  },
  "hardware": {
    "nozzle_min": 200,
    "nozzle_max": 230,
    "speed_min": 30,
    "speed_max": 100,
    "safe_margin": 5.0
  }
}
```

### Aggressive (Exploration)
```json
{
  "learning": {
    "learning_rate": 1e-2,
    "entropy_coef": 0.1,
    "gamma": 0.9
  },
  "hardware": {
    "nozzle_min": 170,
    "nozzle_max": 280,
    "speed_min": 10,
    "speed_max": 200,
    "safe_margin": 15.0
  }
}
```

---

## Configuration Generation Guide

### By Printer Type

**Ender 3 (Budget FDM)**
```python
config.hardware.nozzle_min = 190
config.hardware.nozzle_max = 240
config.hardware.speed_min = 25
config.hardware.speed_max = 150
```

**Prusa i3 (Reliable MK3S)**
```python
config.hardware.nozzle_min = 185
config.hardware.nozzle_max = 245
config.hardware.speed_min = 20
config.hardware.speed_max = 180
```

**Ultimaker 3 (Professional)**
```python
config.hardware.nozzle_min = 185
config.hardware.nozzle_max = 260
config.hardware.speed_min = 15
config.hardware.speed_max = 200
```

---

## Configuration Validation

### Check Configuration Validity
```python
from config import Config

config = Config.load('config.json')
config.validate()  # Raises exception if invalid
```

### Common Validation Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `nozzle_min > nozzle_max` | Inverted limits | Swap values |
| `learning_rate > 0.1` | Too high | Use smaller value |
| `gae_lambda > 1.0` | Invalid range | Use 0.8-0.99 |
| `safe_margin < 0` | Negative margin | Use positive value |

---

## Performance Impact Summary

| Parameter | Impact on Speed | Impact on Quality |
|-----------|-----------------|------------------|
| `learning_rate` | - | ⬆️ (optimal value) |
| `entropy_coef` | - | ⬇️ (too high) |
| `gamma` | - | ⬆️ (higher) |
| `gae_lambda` | - | ⬇️ (too high) |
| `max_grad_norm` | - | ⬆️ (prevents divergence) |
| `nozzle_min/max` | - | ⬆️ (wider range) |
| `speed_min/max` | - | ⬆️ (wider range) |
| `safe_margin` | - | ⬇️ (too high) |
| `curriculum_stages` | ⬆️ | ⬆️ (progressive) |
| `curriculum_steps` | ⬇️ | ⬇️ (too few) |

---

## Configuration Examples by Use Case

### Use Case: Training on Single Printer
```json
{
  "hardware": {"camera_id": 0},
  "federated": {"enabled": false}
}
```

### Use Case: Multi-Printer Lab
```json
{
  "federated": {
    "enabled": true,
    "server_url": "http://lab-server:5001",
    "node_id": "printer_2",
    "sync_frequency": 100
  }
}
```

### Use Case: Production Deployment
```json
{
  "learning": {"learning_rate": 1e-4},
  "hardware": {"safe_margin": 3.0},
  "curriculum": {"curriculum_steps_per_stage": 2000}
}
```

---

## Tips and Tricks

1. **Start Conservative**: Begin with default config, then relax constraints
2. **Monitor Logs**: Watch sovereign.log for signs of problems
3. **A/B Test**: Create two configs and compare results
4. **Document Changes**: Keep notes on why configs were modified
5. **Backup Configs**: Save working configs before modifications
6. **Version Control**: Track config changes in git history

---

For more information, see:
- README.md: Usage guide
- DEVELOPER_GUIDE.md: Architecture and extension
- examples.py: Practical examples
