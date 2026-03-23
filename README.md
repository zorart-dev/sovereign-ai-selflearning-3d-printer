# 🤖 Sovereign-v5.0: Autonomous Edge AI for 3D Printers

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: 40+](https://img.shields.io/badge/Tests-40%2B-brightgreen)](#testing)
[![Documentation: Complete](https://img.shields.io/badge/Documentation-Complete-green)](#documentation)
[![Status: Production Ready](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)](#status)

**A production-ready reinforcement learning system for autonomous control of 3D printers using PPO, LSTM, vision-based rewards, and federated learning.**

## 📋 What Is Sovereign-v5.0?

Sovereign is a complete autonomous AI system that **learns to control 3D printers in real-time**. It uses advanced machine learning to:

✨ **See** - Real-time camera analysis  
🧠 **Think** - Intelligent decision making (PPO + LSTM)  
🎯 **Act** - Autonomous printer control  
🛡️ **Stay Safe** - Maintain safety constraints  
🤝 **Collaborate** - Federated multi-device learning  

### Key Facts
- **2,113 lines** of production code
- **40+ test cases** with comprehensive coverage
- **7 documentation guides** (3,615 lines)
- **10 example scripts** ready to run
- **100% type-hinted** Python
- **MIT License** (free & open source)

---

## 🚀 Quick Start (3 Minutes)

### Step 1: Install
```bash
git clone https://github.com/YOUR_USERNAME/sovereign-v5.0.git
cd sovereign-v5.0
pip install -r requirements.txt
```

### Step 2: Train
```bash
# No hardware needed - uses mock printer
python main.py --printer mock --episodes 10
```

### Step 3: Explore
```bash
# Try all 10 example scripts
python examples.py 1  # Quick start
python examples.py 10 # Batch evaluation
```

**That's it! 🎉 Your AI is learning.**

---

## 📦 Package Contents

### 🧬 Core Implementation (7 modules, 2,113 lines)
- `config.py` - Configuration management
- `networks.py` - Neural architectures (CNN + LSTM)
- `agent.py` - PPO RL agent with GAE
- `reward_safety.py` - Vision rewards & safety
- `hardware.py` - Printer interfaces (3 modes)
- `federated.py` - Distributed learning
- `main.py` - Training orchestration

### 🧪 Testing (40+ tests)
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Error handling tests

### 📚 Documentation (7 guides)
- `README.md` - Overview & quick start (this file)
- `DEVELOPER_GUIDE.md` - Architecture & design
- `CONFIG_GUIDE.md` - Parameter reference
- `QUICK_REFERENCE.md` - Command cheat sheet
- `CONTRIBUTING.md` - How to contribute
- `CHANGELOG.md` - Version history
- `INDEX.md` - Project manifest

### 📄 Setup Files
- `setup.py` - For pip installation
- `requirements.txt` - Dependencies
- `.gitignore` - Git configuration
- `.gitattributes` - File attributes
- `LICENSE` - MIT License

---

## 🤖 How It Works

```
┌──────────────────────────────────────┐
│ PERCEPTION LAYER                      │
│ • Camera input processing             │
│ • Feature extraction (TinyVisionCNN)  │
│ • Sensor normalization                │
└────────────┬─────────────────────────┘
             │
┌────────────▼─────────────────────────┐
│ POLICY LAYER (LSTM Actor-Critic)     │
│ • Temporal memory (LSTM)              │
│ • Action selection (Actor)            │
│ • Value estimation (Critic)           │
└────────────┬─────────────────────────┘
             │
┌────────────▼─────────────────────────┐
│ SAFETY & EXECUTION LAYER              │
│ • Constraint enforcement              │
│ • Hardware control                    │
│ • Failure tracking                    │
│ • Emergency stop                      │
└──────────────────────────────────────┘
```

### Learning Loop
1. **See**: Camera captures printer
2. **Analyze**: Neural network extracts features
3. **Decide**: LSTM chooses best action
4. **Validate**: Safety layer checks constraints
5. **Execute**: Send command to printer
6. **Learn**: PPO updates network weights
7. **Repeat**: 1000s of times to improve

---

## ⚙️ Features

### ✅ Advanced RL
- **PPO**: Stable, proven algorithm
- **Actor-Critic**: Dual network architecture
- **LSTM**: Temporal state memory
- **GAE**: Better advantage estimation
- **Curriculum**: Progressive difficulty

### ✅ Vision & Control
- **Real-time Processing**: OpenCV analysis
- **Quality Metrics**: Edge detection, blob analysis, sharpness
- **Temperature Control**: [180-260]°C automatic
- **Speed Control**: [20-180]mm/min automatic
- **Safety First**: Hard constraints, emergency stop

### ✅ Hardware
- **Mock Printer**: For testing & learning
- **OctoPrint**: For networked printers (Prusa, Creality)
- **Serial**: Direct connection (Marlin firmware)
- **Easy Switching**: Change modes with one parameter

### ✅ Distributed
- **Federated Learning**: Multi-device training
- **Model Aggregation**: Combine learning
- **Checksum Verification**: Data integrity
- **Flexible Sync**: Configure timing

---

## 🔧 Configuration

### Simple
```bash
# Default configuration
python main.py --episodes 100
```

### Custom
```python
from config import Config
config = Config()
config.learning.learning_rate = 1e-4
config.hardware.nozzle_max = 250
config.save('my_config.json')
```

### Command Line
```bash
# High precision
python main.py --config config_precision.json

# Fast training
python main.py --config config_fast.json

# Multi-device
python main.py --federated --episodes 1000
```

---

## 📊 Performance

### Speed
| Component | CPU | GPU |
|-----------|-----|-----|
| Forward pass | 10-50ms | 2-5ms |
| PPO epoch | 500-1000ms | 100-300ms |
| Episode | ~10-20s | ~5-10s |

### Tested Hardware
- ✅ CPU (Python, slow)
- ✅ NVIDIA GPU (10x faster)
- ✅ Apple Silicon (native)
- ✅ AWS/Cloud instances

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Specific test
pytest tests/test_sovereign.py::TestNetworks -v
```

**Coverage**: 40+ tests covering all modules
- Configuration ✅
- Networks ✅
- Agent ✅
- Rewards/Safety ✅
- Hardware ✅
- Federated Learning ✅
- Integration ✅
- Performance ✅

---

## 📚 Documentation

### Read First
- **README.md** (this file) - Overview
- **QUICK_REFERENCE.md** - Common commands

### Then Choose Your Path

**🏃 I want to train quickly**
→ Run `python examples.py 1`

**👨‍💻 I want to understand the code**
→ Read `DEVELOPER_GUIDE.md`

**⚙️ I want to configure parameters**
→ Read `CONFIG_GUIDE.md`

**🤝 I want to contribute**
→ Read `CONTRIBUTING.md`

**🧑‍🔬 I want to modify the algorithm**
→ Study `agent.py` + DEVELOPER_GUIDE.md

---

## 💡 Examples

10 ready-to-run examples:

```bash
python examples.py 1   # Quick start
python examples.py 2   # Resume training
python examples.py 3   # Custom config
python examples.py 4   # Agent usage
python examples.py 5   # Hardware test
python examples.py 6   # Federated setup
python examples.py 7   # Monitoring
python examples.py 8   # Inference
python examples.py 9   # Config generation
python examples.py 10  # Batch eval
```

---

## 🛠️ Hardware Setup

### Mock Printer (No Hardware)
```bash
python main.py --printer mock --episodes 100
```
✅ Perfect for learning and testing

### OctoPrint (Network)
```bash
# 1. Install OctoPrint on Raspberry Pi
# 2. Get API key
# 3. Run:
python main.py --printer octoprint --episodes 100
```
✅ Works with most printers

### Serial/Direct (USB)
```bash
python main.py --printer serial --episodes 100
```
✅ Direct printer connection

---

## 🔐 Safety

### Built-In Protections
- ✅ Hard temperature limits
- ✅ Speed constraints
- ✅ Failure memory
- ✅ Emergency stop
- ✅ Safe margins

### Best Practices
1. Test on mock printer first
2. Use `config_conservative.json`
3. Monitor logs: `tail sovereign.log`
4. Adjust safety margins for your printer
5. Keep checkpoint backups

---

## 📈 What to Expect

### Training Progress
- **Episodes 1-100**: Learning basics
- **Episodes 100-500**: Visible improvement
- **Episodes 500-1000**: Convergence
- **Episodes 1000+**: Refinement

### Monitoring
```bash
# Watch logs in real-time
tail -f sovereign.log

# Check saved models
ls -lh checkpoints/

# View statistics
cat training_log.json | python -m json.tool
```

---

## 🎯 Use Cases

### Home Printing
```bash
python main.py --printer mock --config config_conservative.json
```

### Production
```bash
python main.py --printer octoprint --federated --episodes 5000
```

### Research
```bash
# Modify agent.py and experiment
python examples.py 4  # Direct agent usage
```

### Multi-Device Lab
```bash
# Federated learning
python main.py --federated --episodes 10000
```

---

## 📖 Learning Resources

### In This Repository
- Complete source code with comments
- 40+ test examples
- 10 practical scripts
- 7 comprehensive guides

### External
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [PyTorch Docs](https://pytorch.org/docs/)
- [OpenCV Guide](https://docs.opencv.org/)

---

## ✨ Project Highlights

### Production Grade
- [x] Comprehensive error handling
- [x] Detailed logging
- [x] Safety constraints
- [x] Checkpoint management
- [x] Configuration system

### Well Tested
- [x] 40+ test cases
- [x] All modules covered
- [x] Integration tests
- [x] Performance benchmarks

### Documented
- [x] 7 guides
- [x] 10 examples
- [x] Complete API reference
- [x] Architecture guide

### Extensible
- [x] Clear module separation
- [x] Custom rewards support
- [x] Custom safety rules
- [x] Multiple hardware modes
- [x] Federated learning ready

---

## 🤝 Contributing

Want to contribute? See **CONTRIBUTING.md** for:
- Code standards
- Testing requirements
- Pull request process
- Issue reporting

---

## 📄 License

MIT License - Free to use, modify, and distribute

---

## 📞 Support

### Need Help?
1. Check README (this file)
2. Try examples: `python examples.py <1-10>`
3. Read DEVELOPER_GUIDE.md
4. Check QUICK_REFERENCE.md
5. Review test examples
6. Create GitHub issue

### Documentation Files
- README.md (overview)
- DEVELOPER_GUIDE.md (deep dive)
- CONFIG_GUIDE.md (parameters)
- QUICK_REFERENCE.md (commands)
- examples.py (working code)
- tests/ (usage patterns)

---

## 🎊 Getting Started Now

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/sovereign-v5.0.git
cd sovereign-v5.0

# 2. Install
pip install -r requirements.txt

# 3. Train
python main.py --printer mock --episodes 10

# 4. Explore
python examples.py 1

# Done! 🚀
```

---

**Made with ❤️ for the 3D printing community**

Sovereign-v5.0 • PPO + LSTM • Vision-Guided • Safety-First • Open Source
