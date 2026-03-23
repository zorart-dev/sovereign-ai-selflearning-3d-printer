# Sovereign-v5.0 Quick Reference

Fast lookup for common commands, patterns, and troubleshooting.

## Command Line Cheat Sheet

### Basic Commands
```bash
# Quick start
python main.py --printer mock --episodes 10

# Full training
python main.py --episodes 1000

# Resume training
python main.py --checkpoint checkpoints/best_model.pt --episodes 500

# Custom config
python main.py --config my_config.json

# With federated learning
python main.py --federated --printer octoprint --episodes 1000

# With specific printer
python main.py --printer serial --episodes 100
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_sovereign.py::TestNetworks -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Run example scripts
python examples.py 1  # Quick start example
python examples.py 4  # Agent usage
python examples.py 10 # Batch evaluation
```

### Code Quality
```bash
# Format code
black sovereign_v5_final/

# Check style
flake8 sovereign_v5_final/

# Type checking
mypy sovereign_v5_final/

# Full validation
black sovereign_v5_final/ && flake8 sovereign_v5_final/
```

---

## Python API Quick Reference

### Import Statements
```python
from config import Config
from networks import ActorCriticModel
from agent import SovereignAgent
from main import SovereignAutonomousSystem
from hardware import create_printer, MockPrinter
from reward_safety import AdaptiveSafety, TemporalVisionReward
from federated import FederatedLearningNode
import torch
import numpy as np
```

### Create System
```python
# Simple
system = SovereignAutonomousSystem()

# Full config
system = SovereignAutonomousSystem(
    config_path='config.json',
    printer_mode='mock',
    use_federated=False,
    checkpoint_path='checkpoints/best.pt'
)

# Train
system.train(num_episodes=100, save_interval=10)
system.shutdown()
```

### Load/Save Configuration
```python
# Load
config = Config.load('config.json')

# Modify
config.learning.learning_rate = 1e-4
config.hardware.nozzle_max = 250

# Save
config.save('custom_config.json')

# Validate
config.validate()
```

### Work with Models
```python
# Create
config = Config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ActorCriticModel(config).to(device)

# Create agent
agent = SovereignAgent(model, config, device)

# Action
action = agent.act(state)

# Training
agent.store_transition(state, action, reward)
agent.compute_gae(next_state)
stats = agent.train_ppo_step()

# Save/Load
agent.save_checkpoint('model.pt')
agent.load_checkpoint('model.pt')
```

### Hardware Control
```python
# Create printer
printer = create_printer('mock')

# Get state
state = printer.get_state()
temp = state.nozzle_temp
speed = state.extrusion_speed

# Control
printer.set_temperature(220, 60)
printer.set_extrusion_speed(100)
printer.emergency_stop()

# Check connection
if printer.is_connected():
    print("Ready")
```

### Reward and Safety
```python
# Rewards
reward = TemporalVisionReward(config)
r = reward.get()  # Returns float in [-1, 1]
reward.close()

# Safety
safety = AdaptiveSafety(config)
is_safe, cmd = safety.validate(action, temp, speed)
safety.report_failure(temp)
stats = safety.get_stats()
```

### Federated Learning
```python
# Node
node = FederatedLearningNode('printer_1', server_url='http://server:5001')

if node.should_sync():
    node.upload_local_model(model)
    node.download_global_model(model)
    node.record_sync()

# Server
server = FederatedServer(port=5001)
server.receive_model('printer_1', model_data)
server.aggregate_models()
```

---

## Configuration Snippets

### Minimal Config (Testing)
```json
{
  "learning": {"learning_rate": 3e-4},
  "hardware": {"camera_id": 0}
}
```

### High-Performance (GPU)
```json
{
  "learning": {
    "learning_rate": 1e-3,
    "batch_size": 64,
    "sequence_length": 16
  }
}
```

### Low-Resource (CPU/Edge)
```json
{
  "learning": {
    "learning_rate": 1e-4,
    "batch_size": 16,
    "sequence_length": 8
  }
}
```

### Safe Production
```json
{
  "hardware": {
    "nozzle_min": 190,
    "nozzle_max": 230,
    "safe_margin": 3.0
  }
}
```

---

## File Structure Reference

```
sovereign_v5_final/
├── config.py              # Configuration classes
├── networks.py            # Neural networks
├── agent.py               # RL agent (PPO)
├── reward_safety.py       # Rewards and safety
├── hardware.py            # Hardware interfaces
├── federated.py           # Federated learning
├── main.py                # Training loop
├── examples.py            # Example scripts
├── setup.py               # Installation
├── requirements.txt       # Dependencies
├── __init__.py            # Package init
│
├── README.md              # User guide
├── DEVELOPER_GUIDE.md     # Developer guide
├── CONFIG_GUIDE.md        # Configuration reference
├── BUILD_REPORT.md        # Build status
│
├── tests/
│   ├── __init__.py
│   └── test_sovereign.py  # Test suite
│
├── checkpoints/           # Saved models (auto-created)
│   ├── best_model.pt
│   ├── episode_10.pt
│   └── final_model.pt
│
└── .gitignore
```

---

## Common Tasks

### Task: Train from Scratch
```python
system = SovereignAutonomousSystem(printer_mode='mock')
system.train(num_episodes=100)
```

### Task: Continue Training
```python
system = SovereignAutonomousSystem(
    checkpoint_path='checkpoints/best_model.pt'
)
system.train(num_episodes=100)
```

### Task: Evaluate Model
```python
system = SovereignAutonomousSystem(
    checkpoint_path='checkpoints/best_model.pt'
)
for ep in range(10):
    ret, stats = system.collect_trajectory()
    print(f"Return: {ret:.3f}")
```

### Task: Change Configuration
```python
config = Config()
config.learning.learning_rate = 5e-3
config.hardware.nozzle_max = 250
config.save('my_config.json')

system = SovereignAutonomousSystem(config_path='my_config.json')
system.train(num_episodes=50)
```

### Task: Test Hardware
```python
printer = create_printer('mock')
state = printer.get_state()
print(f"Temp: {state.nozzle_temp:.1f}°C")
```

### Task: Federated Setup
```python
node = FederatedLearningNode(
    'printer_1',
    server_url='http://192.168.1.100:5001'
)
system = SovereignAutonomousSystem(use_federated=True)
system.train(num_episodes=1000)
```

---

## Troubleshooting Quick Fixes

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Reduce batch_size in config |
| NaN in loss | Reduce learning_rate by 10x |
| Training too slow | Check GPU utilization, use `--printer mock` |
| Safety violations | Increase safe_margin or lower nozzle_max |
| Camera not found | Check camera_id with `cv2.VideoCapture(id)` |
| OctoPrint connection failed | Verify URL and API key in config |
| Checkpoint won't load | Check file path and version compatibility |
| Tests fail to import | Run `pip install -r requirements.txt` |

---

## Key Metrics to Monitor

**Training Progress**:
```
Episode Return: target 0-5, should increase over time
Mean Reward: target >0, indicates progress
Episode Length: target 100-200 steps
```

**Safety**:
```
Total Failures: should be 0 or very low
Consecutive Failures: should never exceed 4
Emergency Stop: should never activate
```

**Performance**:
```
Training Speed: check sovereign.log timestamps
Policy Loss: should decrease over time
Value Loss: should decrease over time
```

---

## Performance Benchmarks

| Component | Time | Memory |
|-----------|------|--------|
| Forward Pass (CPU) | 10-50ms | ~50MB |
| Forward Pass (GPU) | 2-5ms | ~200MB |
| PPO Epoch | 500-1000ms | ~2GB |
| Episode (200 steps) | ~10-20s | ~500MB |
| Full Training (1000 eps) | ~10-50 hours | Peak: 4GB |

---

## Environment Variables

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Disable CUDA (force CPU)
export CUDA_VISIBLE_DEVICES=

# Reduce PyTorch threads
export OMP_NUM_THREADS=4

# Run
python main.py --episodes 100
```

---

## Git Commands

```bash
# Initialize repo
git init
git add .
git commit -m "Initial commit: Sovereign-v5.0"

# Create branch for experiments
git checkout -b experiment-high-lr

# Revert changes
git checkout -- config.json

# View changes
git diff config.json

# Stash experiments
git stash
git stash pop
```

---

## Python Snippets

### Quick Test
```python
from config import Config
from networks import ActorCriticModel
import torch

config = Config()
model = ActorCriticModel(config)
x_vision = torch.randn(1, 3, 240, 320)
x_sensors = torch.randn(1, 8)
action, logprob, value = model(x_vision, x_sensors)
print(f"Action: {action}, Value: {value:.3f}")
```

### Check GPU
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Profile Code
```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
system = SovereignAutonomousSystem()
system.train_episode()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

---

## Documentation Links

- **Usage**: See README.md
- **Architecture**: See DEVELOPER_GUIDE.md
- **Configuration**: See CONFIG_GUIDE.md
- **Examples**: Run `python examples.py <1-10>`
- **Tests**: See tests/test_sovereign.py
- **Build Info**: See BUILD_REPORT.md

---

## Quick Help Commands

```bash
# Show help
python main.py --help

# List examples
python examples.py

# Run specific example
python examples.py 1

# Test single component
pytest tests/test_sovereign.py::TestNetworks -v

# View logs
tail -f sovereign.log

# Check training progress
tail -20 training_log.json | python -m json.tool

# Show available cameras
python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
```

---

## Emergency Commands

```bash
# Kill stuck process
pkill -f "python main.py"

# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Reset state
rm sovereign.log training_log.json

# Backup training
cp -r checkpoints checkpoints.backup

# Restore backup
cp -r checkpoints.backup/* checkpoints/
```

---

**Last Updated**: 2026-03-23
**For Details**: See full documentation in README.md and DEVELOPER_GUIDE.md
