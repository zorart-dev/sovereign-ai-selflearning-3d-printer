# GitHub Quick Start Guide

Start using Sovereign-v5.0 in 5 minutes!

---

## 🚀 5-Minute Quick Start

### Step 1: Clone Repository (1 minute)
```bash
git clone https://github.com/USERNAME/sovereign-v5.0.git
cd sovereign-v5.0
```

### Step 2: Install (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 3: Run Example (2 minutes)
```bash
python main.py --printer mock --episodes 5
```

### Done! 🎉

Check the logs:
```bash
tail sovereign.log
```

---

## 📚 Documentation Structure

**Start Here:**
- 🎯 **WHAT_IS_IT.md** - Understand what Sovereign does
- 📖 **HOW_TO.md** - Complete usage guide  
- ⚡ **README.md** - Features and quick start

**For Different Needs:**
- 💻 **DEVELOPER_GUIDE.md** - Architecture and internals
- ⚙️ **CONFIG_GUIDE.md** - Configuration parameters
- ⌨️ **QUICK_REFERENCE.md** - Command cheat sheet

**For Contributors:**
- 🤝 **CONTRIBUTING.md** - How to contribute
- 📋 **CHANGELOG.md** - Version history

---

## 🎯 Choose Your Path

### Path 1: I want to understand what this is (10 min)
→ Read: **WHAT_IS_IT.md**
- What is Sovereign?
- How does it work?
- What are the benefits?

### Path 2: I want to try it now (15 min)
→ Run: `python examples.py 1`
→ Read: **HOW_TO.md** (Quick Start section)

### Path 3: I want to use it on my printer (30 min)
→ Follow: **HOW_TO.md** (Hardware Setup section)
→ Check: **CONFIG_GUIDE.md** for your printer

### Path 4: I want to modify/extend it (1 hour+)
→ Study: **DEVELOPER_GUIDE.md**
→ Review: Source code in main modules
→ Check: tests/ for examples

### Path 5: I want to contribute (ongoing)
→ Read: **CONTRIBUTING.md**
→ Look at: existing issues & PRs
→ Start small!

---

## 🎓 Learning Resources

### Inside This Repository
| File | Purpose | Read Time |
|------|---------|-----------|
| WHAT_IS_IT.md | Understand the concept | 15 min |
| HOW_TO.md | Complete usage guide | 30 min |
| README.md | Features & quick start | 10 min |
| CONFIG_GUIDE.md | Parameter reference | 20 min |
| DEVELOPER_GUIDE.md | Architecture details | 40 min |
| QUICK_REFERENCE.md | Command cheat sheet | 5 min |
| CONTRIBUTING.md | Contribution guide | 10 min |
| examples.py | 10 working examples | varies |

### Total Time Investment
- Quick overview: **20 minutes**
- Get it running: **30 minutes**
- Full understanding: **2-3 hours**
- Deep expertise: **1-2 weeks**

---

## 📦 What's Included

### Code
- ✅ 7 production-ready modules
- ✅ 2,113 lines of implementation
- ✅ 100% type hints
- ✅ Full docstrings

### Tests
- ✅ 35+ comprehensive tests
- ✅ All modules covered
- ✅ Run with: `pytest tests/ -v`

### Examples
- ✅ 10 ready-to-run scripts
- ✅ Quick start to advanced
- ✅ Run with: `python examples.py <1-10>`

### Documentation
- ✅ 7 comprehensive guides
- ✅ 3,600+ lines of docs
- ✅ Real-world examples

---

## 💻 System Requirements

### Minimum
- Python 3.9+
- 2GB RAM
- 500MB disk space
- CPU works fine

### Recommended
- Python 3.10+
- 8GB RAM
- GPU with 4GB+ VRAM
- SSD for faster I/O

### Optional
- USB camera (for vision features)
- OctoPrint server (for real printers)

---

## 🚀 Common Tasks

### Run a Quick Test
```bash
python examples.py 1
```

### Try a Specific Example
```bash
python examples.py 4  # Direct agent usage
python examples.py 7  # Monitoring
python examples.py 10 # Batch evaluation
```

### Train on Your Hardware
```bash
python main.py --printer octoprint --episodes 100
```

### Resume Previous Training
```bash
python main.py --checkpoint checkpoints/best_model.pt --episodes 500
```

### Check Documentation
```bash
# Quick reference
less QUICK_REFERENCE.md

# Configuration help
less CONFIG_GUIDE.md

# Architecture
less DEVELOPER_GUIDE.md
```

### Run Tests
```bash
pytest tests/ -v
pytest tests/ --cov=.  # With coverage
```

---

## ❓ FAQ

**Q: Is this stable/production-ready?**  
A: Yes! Production-grade code with comprehensive testing.

**Q: Will it work with my printer?**  
A: If it supports OctoPrint or serial, yes. See HOW_TO.md.

**Q: How long does it take to learn?**  
A: Quick demo: 5 min. Basic usage: 30 min. Full mastery: 2-3 hours.

**Q: Do I need a GPU?**  
A: No, but it's 5-10x faster with GPU.

**Q: Can I use this commercially?**  
A: Yes! MIT license - full commercial freedom.

**Q: Where do I get help?**  
A: Check documentation, run examples, or file an issue.

---

## 🆘 Troubleshooting

### Installation Issues
```bash
# Check Python version
python --version  # Should be 3.9+

# Verify packages
python -c "import torch, cv2, numpy; print('OK')"

# Reinstall if needed
pip install -r requirements.txt --upgrade
```

### Can't Import Modules
```bash
# Test imports
python -c "from config import Config; print('OK')"
python -c "from networks import ActorCriticModel; print('OK')"
python -c "from agent import SovereignAgent; print('OK')"
```

### Performance Issues
```bash
# Check if GPU available
python -c "import torch; print(torch.cuda.is_available())"

# If slow, try GPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Camera Not Working
```bash
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i} found')
        cap.release()
"
```

---

## 📞 Getting Help

### Documentation
1. Check the relevant guide (HOW_TO, CONFIG_GUIDE, etc.)
2. Search QUICK_REFERENCE.md
3. Review examples.py

### Code Examples
- Look in `examples.py` (10 working examples)
- Check `tests/` for usage patterns
- Review docstrings in source code

### Online
- GitHub Issues - Report bugs or ask questions
- GitHub Discussions - Community help
- README.md - Common issues & solutions

### Contributing
- See CONTRIBUTING.md for guidelines
- Start with small PRs
- Help others!

---

## 📈 Next Steps

### After Quick Start
1. ✅ Understand what it does (WHAT_IS_IT.md)
2. ✅ Learn how to use it (HOW_TO.md)
3. ✅ Run the examples
4. ✅ Customize for your setup

### After Basics
1. Read CONFIG_GUIDE.md
2. Try different configurations
3. Train for longer
4. Monitor progress
5. Experiment!

### After Mastery
1. Study DEVELOPER_GUIDE.md
2. Review source code
3. Create extensions
4. Contribute back!

---

## 🎯 Use Case Examples

### For Makers
```bash
# Optimize print quality on your printer
python main.py --printer octoprint --episodes 100
```

### For Researchers
```bash
# Study PPO implementation
less agent.py          # See PPO algorithm
python examples.py 4   # Direct agent usage
```

### For Developers
```bash
# Build custom extensions
python -c "from config import Config; from networks import ActorCriticModel"
# Now implement your custom classes
```

### For Teams
```bash
# Use distributed training
python main.py --federated --episodes 1000
```

---

## 🔒 Safety First

Sovereign includes multiple safety layers:
- ✅ Hard parameter constraints
- ✅ Failure detection & learning
- ✅ Emergency stop mechanism
- ✅ Real-time monitoring

**It's designed to be safe from the start.**

---

## 📊 What You Can Expect

### After 10 Episodes
- System is learning
- Parameters being adjusted
- Safety constraints working

### After 50 Episodes
- Clear improvement pattern
- Optimal range emerging
- Error rate decreasing

### After 100 Episodes
- Stable convergence
- Near-optimal parameters found
- Ready for production use

### After 500+ Episodes
- Fine-tuning happening
- Excellent print quality
- Minimal parameter variation

---

## 💡 Pro Tips

1. **Start with mock printer** - Safer and faster
2. **Monitor logs** - `tail -f sovereign.log`
3. **Save configs** - Create configs for different filaments
4. **Keep backups** - Copy good checkpoints
5. **Start conservative** - Increase complexity gradually
6. **Read examples** - Learn from working code
7. **Join community** - Share your results!

---

## 🎉 You're Ready to Begin!

Choose a path above and start your Sovereign journey:

- 🎯 **5 minutes**: Run `python main.py --printer mock --episodes 5`
- 📖 **10 minutes**: Read WHAT_IS_IT.md
- 💻 **30 minutes**: Read HOW_TO.md and try examples
- 🧠 **2 hours**: Study all documentation
- 🚀 **Ongoing**: Build amazing things!

---

## 📚 Complete Documentation Index

| Purpose | Document | Time |
|---------|----------|------|
| What is this? | WHAT_IS_IT.md | 15 min |
| How do I use it? | HOW_TO.md | 30 min |
| Quick reference | QUICK_REFERENCE.md | 5 min |
| Configuration help | CONFIG_GUIDE.md | 20 min |
| Architecture details | DEVELOPER_GUIDE.md | 40 min |
| Want to help? | CONTRIBUTING.md | 10 min |
| What's new? | CHANGELOG.md | 5 min |
| Full details | README.md | 15 min |

---

**Happy coding! 🚀**

Questions? Check the relevant documentation above or file an issue.

Ready to teach your 3D printer to be smarter?
