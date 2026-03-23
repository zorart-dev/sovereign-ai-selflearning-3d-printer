# How To: Sovereign-v5.0 Complete Guide

Comprehensive instructions for every aspect of using Sovereign-v5.0.

---

## 📥 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 2GB RAM minimum (4GB+ recommended)
- GPU optional but recommended

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Required packages:**
- torch >= 2.0.0 (PyTorch)
- torchvision (computer vision)
- numpy (numerical computing)
- opencv-python (camera & image processing)
- requests (API communication)

### Step 2: Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

All should print versions without errors.

---

## 🚀 Quick Start

### Option 1: Mock Printer (Recommended for First Time)
```bash
python main.py --printer mock --episodes 10
```

This trains the AI on a simulated printer for 10 episodes. Good for testing.

### Option 2: OctoPrint (Real Networked Printer)
```bash
python main.py --printer octoprint --episodes 100
```

Requires OctoPrint to be running and accessible.

### Option 3: Serial (Direct Connection)
```bash
python main.py --printer serial --episodes 100
```

For direct serial connection (Marlin-compatible).

---

## 🎮 Running Examples

Sovereign includes 10 ready-to-run example scripts.

### List Available Examples
```bash
python examples.py
```

### Run Specific Example
```bash
python examples.py 1      # Quick start
python examples.py 2      # Resume training
python examples.py 3      # Custom config
python examples.py 4      # Agent direct usage
python examples.py 5      # Hardware testing
python examples.py 6      # Federated learning
python examples.py 7      # Monitoring metrics
python examples.py 8      # Inference mode
python examples.py 9      # Config generation
python examples.py 10     # Batch evaluation
```

### Example 1: Quick Start (5 minutes)
```bash
python examples.py 1
```
- Trains for 10 episodes
- Uses mock printer
- Shows basic functionality
- Good for first-time users

### Example 4: Direct Agent Usage
```bash
python examples.py 4
```
- Low-level API usage
- Manual training loops
- Direct model access
- For advanced users

---

## ⚙️ Configuration

### Using Default Configuration
```bash
python main.py
```
Uses built-in defaults.

### Using Custom Configuration
```bash
# Create custom config
python -c "
from config import Config
config = Config()
config.learning.learning_rate = 1e-3
config.hardware.nozzle_max = 250
config.save('my_config.json')
"

# Run with custom config
python main.py --config my_config.json --episodes 100
```

### Configuration Options

**Learning Parameters:**
- `learning_rate`: 1e-4 to 1e-2 (default: 3e-4)
- `entropy_coef`: 0 to 0.1 (default: 0.01)
- `gamma`: 0.9 to 0.999 (default: 0.99)

**Hardware Constraints:**
- `nozzle_min` / `nozzle_max`: Temperature range in °C
- `speed_min` / `speed_max`: Speed range in mm/min
- `safe_margin`: Margin around unsafe zones

**Curriculum Learning:**
- `curriculum_stages`: Reward scaling per stage
- `curriculum_steps_per_stage`: Steps per stage

See CONFIG_GUIDE.md for complete reference.

---

## 🔧 Hardware Setup

### Mock Printer (No Hardware Needed)
```bash
python main.py --printer mock
```
Perfect for testing and development.

### OctoPrint Setup

**1. Install OctoPrint**
```bash
pip install octoprint
```

**2. Start OctoPrint**
```bash
octoprint serve
```

**3. Get API Key**
- Open http://localhost:5000
- Settings → API
- Copy API key

**4. Run Sovereign**
```bash
python main.py --printer octoprint \
  --config my_config.json \
  --episodes 100
```

Update config.json with:
```json
{
  "federated": {
    "enabled": false
  }
}
```

### Serial Printer Setup

**1. Connect Printer**
```bash
ls /dev/ttyUSB*  # Linux
COM3            # Windows
```

**2. Run Sovereign**
```bash
python main.py --printer serial --episodes 100
```

---

## 📊 Monitoring Training

### Watch Live Logs
```bash
tail -f sovereign.log
```

### Check Statistics
```bash
cat training_log.json | python -m json.tool | head -50
```

### Plot Progress
```bash
python -c "
import json
with open('training_log.json') as f:
    logs = json.load(f)
    returns = [log['episode_return'] for log in logs]
    print(f'Mean Return: {sum(returns) / len(returns):.3f}')
    print(f'Max Return: {max(returns):.3f}')
"
```

### Key Metrics to Watch
- **Episode Return**: Should increase over time
- **Mean Reward**: Target > 0
- **Episode Length**: Typical 100-200 steps
- **Safety Violations**: Should be 0 or very low

---

## 🧪 Testing

### Run Full Test Suite
```bash
pytest tests/ -v
```

### Run Specific Test
```bash
pytest tests/test_sovereign.py::TestNetworks -v
```

### Run with Coverage
```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

### Custom Test Run
```bash
python -c "
from tests.test_sovereign import TestMockPrinter
test = TestMockPrinter()
test.test_mock_printer_creation()
print('✓ Test passed')
"
```

---

## 💾 Checkpoints

### Resume Training
```bash
python main.py --checkpoint checkpoints/best_model.pt --episodes 500
```

### List Available Checkpoints
```bash
ls -lh checkpoints/
```

### Load and Evaluate
```bash
python examples.py 8  # Inference mode
```

### Save Custom Checkpoint
```bash
python -c "
from agent import SovereignAgent
from networks import ActorCriticModel
from config import Config
import torch

config = Config()
model = ActorCriticModel(config)
agent = SovereignAgent(model, config, torch.device('cpu'))

# ... train ...

agent.save_checkpoint('my_checkpoint.pt')
"
```

---

## 🎯 Advanced Usage

### Custom Reward Function
```python
from reward_safety import TemporalVisionReward

class CustomReward(TemporalVisionReward):
    def get(self):
        # Your custom logic
        return custom_reward_value

# Use in system
system = SovereignAutonomousSystem()
system.reward = CustomReward(system.config)
```

### Custom Safety Constraints
```python
from reward_safety import AdaptiveSafety

class CustomSafety(AdaptiveSafety):
    def validate(self, action_id, temp, speed):
        # Your custom validation
        return is_safe, command

# Use in system
system = SovereignAutonomousSystem()
system.safety = CustomSafety(system.config)
```

### Distributed Training (Federated Learning)
```bash
python main.py --federated --episodes 1000
```

Requires server running (see DEVELOPER_GUIDE.md).

---

## 🔍 Debugging

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check GPU
```bash
python -c "
import torch
print('CUDA Available:', torch.cuda.is_available())
print('Device:', torch.cuda.current_device() if torch.cuda.is_available() else 'CPU')
print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
"
```

### Profile Code
```bash
python -m cProfile -s cumulative main.py --printer mock --episodes 2
```

### Check Imports
```bash
python -c "
import config, networks, agent, reward_safety, hardware, federated, main
print('✓ All modules imported successfully')
"
```

---

## 🐛 Common Issues & Solutions

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in config or use CPU
```json
{
  "learning": {
    "batch_size": 32
  }
}
```

### Issue: Camera Not Found
**Solution**: Check camera ID
```bash
python -c "
import cv2
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} available')
"
```

### Issue: NaN in Loss
**Solution**: Reduce learning rate
```bash
python main.py --config my_config.json  # with lower lr
```

### Issue: Training Too Slow
**Solution**: Use GPU
```bash
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

### Issue: Safety Violations
**Solution**: Increase safe margin
```json
{
  "hardware": {
    "safe_margin": 10.0
  }
}
```

---

## 📈 Performance Tuning

### Fast Convergence
```json
{
  "learning": {
    "learning_rate": 1e-2,
    "entropy_coef": 0.05
  },
  "hardware": {
    "safe_margin": 10.0
  }
}
```

### High Precision
```json
{
  "learning": {
    "learning_rate": 1e-4,
    "entropy_coef": 0.001
  },
  "hardware": {
    "safe_margin": 3.0
  }
}
```

### Conservative (Safety-First)
```json
{
  "learning": {
    "learning_rate": 1e-5
  },
  "hardware": {
    "nozzle_min": 200,
    "nozzle_max": 230,
    "safe_margin": 5.0
  }
}
```

---

## 🔗 Integration

### With OctoPrint
```python
from hardware import create_printer
printer = create_printer('octoprint', 
    url='http://192.168.1.100:5000',
    api_key='YOUR_KEY')
```

### With Custom Hardware
```python
from hardware import PrinterInterface

class MyPrinter(PrinterInterface):
    def get_state(self):
        # Read state
        pass
    
    def set_temperature(self, nozzle, bed):
        # Set temps
        pass
    
    # Implement other methods
```

---

## 📚 Learning Paths

### Path 1: Quick Learner (1 hour)
1. Read: WHAT_IS_IT.md (10 min)
2. Install: pip install (5 min)
3. Run: `python examples.py 1` (5 min)
4. Experiment: Try other examples (40 min)

### Path 2: Hands-On (2 hours)
1. Read: README.md (15 min)
2. Install: Full setup (10 min)
3. Run: Quick start example (5 min)
4. Customize: Modify config (15 min)
5. Monitor: Watch training (30 min)
6. Experiment: Try variations (45 min)

### Path 3: Deep Dive (4 hours)
1. Read: All documentation (1 hour)
2. Study: Code modules (1 hour)
3. Run: All examples (1 hour)
4. Implement: Custom features (1 hour)

### Path 4: Research (8+ hours)
1. Study: DEVELOPER_GUIDE.md (1 hour)
2. Review: Algorithm implementations (2 hours)
3. Experiment: Variations (2 hours)
4. Implement: Novel ideas (3+ hours)

---

## 🚢 Deployment

### Development Deployment
```bash
python main.py --printer mock --episodes 10
```

### Testing Deployment
```bash
python main.py --printer octoprint --episodes 100
```

### Production Deployment
```bash
# 1. Create optimized config
python -c "from config import Config; c = Config(); c.save('prod_config.json')"

# 2. Train on real hardware
python main.py --config prod_config.json --episodes 1000

# 3. Backup best model
cp checkpoints/best_model.pt backup_model.pt

# 4. Monitor continuously
python -c "
import json, time
while True:
    with open('training_log.json') as f:
        logs = json.load(f)
    print(f'Latest: {logs[-1][\"episode_return\"]:.3f}')
    time.sleep(60)
"
```

---

## 🔐 Best Practices

### Configuration
- ✅ Test with mock printer first
- ✅ Start with default settings
- ✅ Adjust gradually (10% at a time)
- ✅ Monitor for safety violations
- ✅ Backup working configurations

### Training
- ✅ Start with short training (10-50 episodes)
- ✅ Monitor logs during training
- ✅ Check checkpoints regularly
- ✅ Save best models
- ✅ Create backups

### Hardware
- ✅ Test connection before training
- ✅ Verify printer compatibility
- ✅ Check camera quality
- ✅ Ensure safe operating conditions
- ✅ Have manual controls available

### Safety
- ✅ Always monitor first run
- ✅ Have emergency stop ready
- ✅ Verify constraints match hardware
- ✅ Test with low learning rates first
- ✅ Increase complexity gradually

---

## 🎓 Learning Resources

**Inside Package:**
- README.md - Overview & quick start
- CONFIG_GUIDE.md - Parameter reference
- DEVELOPER_GUIDE.md - Architecture details
- QUICK_REFERENCE.md - Cheat sheet
- examples.py - 10 working examples

**Online Resources:**
- PyTorch Docs: pytorch.org/docs
- OpenCV Guide: docs.opencv.org
- OctoPrint API: docs.octoprint.org

---

## 📞 Getting Help

### Check Documentation
1. README.md - Features and usage
2. QUICK_REFERENCE.md - Commands
3. CONFIG_GUIDE.md - Parameters
4. DEVELOPER_GUIDE.md - Architecture

### Run Examples
```bash
python examples.py <1-10>
```

### Check Logs
```bash
tail -100 sovereign.log
```

### Run Tests
```bash
pytest tests/ -v
```

---

## ✅ Checklist for First Run

- [ ] Python 3.9+ installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Tests pass: `pytest tests/test_sovereign.py::TestConfig -v`
- [ ] Camera working (if using vision): `python -c "import cv2; cv2.VideoCapture(0)"`
- [ ] Config valid: `python -c "from config import Config; Config().validate()"`
- [ ] Mock example works: `python examples.py 1`
- [ ] Logs created: `ls -l sovereign.log`
- [ ] Checkpoint saved: `ls checkpoints/`
- [ ] Training metrics exported: `cat training_log.json`

---

## 🎉 You're Ready!

You now have everything you need to:
- ✅ Install Sovereign
- ✅ Run examples
- ✅ Configure for your hardware
- ✅ Train and monitor
- ✅ Debug issues
- ✅ Deploy to production

**Next Steps:**
1. Run a quick example: `python examples.py 1`
2. Explore the code
3. Customize for your printer
4. Start training!

Happy coding! 🚀
