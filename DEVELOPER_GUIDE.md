"""
DEVELOPER GUIDE: Sovereign-v5.0

This guide covers:
- Architecture and design decisions
- Extension points and customization
- Advanced configuration
- Performance optimization
- Debugging and troubleshooting
"""

# ============================================================================
# ARCHITECTURE AND DESIGN
# ============================================================================

"""
1. SYSTEM ARCHITECTURE

The system is organized into distinct layers:

┌─────────────────────────────────────────────────────────────┐
│                     TRAINING ORCHESTRATOR                  │
│              (SovereignAutonomousSystem in main.py)        │
├─────────────────────────────────────────────────────────────┤
│                    CORE RL AGENT LAYER                      │
│  (Networks + Agent: Policy learning, GAE, PPO updates)     │
├─────────────────────────────────────────────────────────────┤
│           PERCEPTION & FEEDBACK LAYER                       │
│  (Rewards, Vision, Safety Constraints)                      │
├─────────────────────────────────────────────────────────────┤
│              HARDWARE ABSTRACTION LAYER                      │
│  (Printer Interface: Mock/OctoPrint/Serial)                │
├─────────────────────────────────────────────────────────────┤
│         SUPPORTING INFRASTRUCTURE LAYER                     │
│  (Federated Learning, Config, Logging, Checkpointing)      │
└─────────────────────────────────────────────────────────────┘


2. MODULE RESPONSIBILITIES

config.py
  - Central configuration management
  - Parameter validation
  - Default values for all components
  - Config serialization/deserialization

networks.py
  - Neural network architectures
  - Feature extraction (TinyVisionCNN)
  - Policy network (LSTMActorCritic)
  - Value function (Critic head)

agent.py
  - RL algorithm implementation (PPO)
  - Experience collection
  - Gradient-based learning (GAE + PPO updates)
  - Model persistence

reward_safety.py
  - Vision-based reward calculation
  - Safety constraint enforcement
  - Failure tracking and emergency stop

hardware.py
  - Printer interface abstraction
  - Hardware mode selection
  - State management
  - Command execution

federated.py
  - Distributed learning coordination
  - Model aggregation
  - Synchronization

main.py
  - Training loop orchestration
  - Component integration
  - Curriculum management
  - Logging and checkpointing


3. DATA FLOW

Episode Flow:
  1. SovereignAutonomousSystem.train_episode()
  2. collect_trajectory() → camera frame + sensor state
  3. agent.act() → policy selects action (0-3)
  4. safety.validate() → constraint checking
  5. hardware.execute() → send command to printer
  6. reward.get() → vision-based reward computation
  7. agent.store_transition() → add to replay buffer
  8. agent.compute_gae() → advantage estimation
  9. agent.train_ppo_step() → gradient update
  10. Results logged and checkpointed


4. DESIGN PATTERNS

Singleton-like Config:
  config = Config.load('config.json')  # Single source of truth
  
Factory Pattern for Hardware:
  printer = create_printer('mock')
  printer = create_printer('octoprint', url='...')
  
Strategy Pattern for Reward Calculation:
  class CustomReward:
      def get(self) -> float: ...
  
Context Manager for Resources:
  with torch.no_grad():
      action = agent.act(state)


# ============================================================================
# EXTENSION POINTS
# ============================================================================

"""

Extending Sovereign-v5.0:

1. CUSTOM REWARD FUNCTION
   
   Inherit from TemporalVisionReward:
   
   class CustomReward(TemporalVisionReward):
       def _analyze_frame(self, frame):
           # Your custom analysis
           return {'quality': ..., 'stability': ...}
       
       def get(self) -> float:
           # Your reward calculation
           return custom_reward_value
   
   Usage:
       reward = CustomReward(config)
       system.reward = reward


2. CUSTOM HARDWARE INTERFACE
   
   Inherit from PrinterInterface:
   
   class CustomPrinter(PrinterInterface):
       def get_state(self) -> PrinterState:
           # Your state reading
           pass
       
       def set_temperature(self, nozzle, bed) -> bool:
           # Your temperature control
           pass
   
   Usage:
       printer = CustomPrinter()
       system.printer = printer


3. CUSTOM NEURAL NETWORK
   
   Inherit from torch.nn.Module:
   
   class CustomPolicy(torch.nn.Module):
       def __init__(self, config):
           super().__init__()
           self.perception = CustomPerception(config)
           self.policy = CustomActor(config)
           self.critic = CustomCritic(config)
       
       def forward(self, vision, sensors):
           features = self.perception(vision, sensors)
           action = self.policy(features)
           value = self.critic(features)
           return action, log_prob, value
   
   Usage:
       model = CustomPolicy(config).to(device)
       agent = SovereignAgent(model, config, device)


4. CUSTOM TRAINING ALGORITHM
   
   Inherit from SovereignAgent:
   
   class CustomAgent(SovereignAgent):
       def train_ppo_step(self):
           # Your custom training logic
           # Can use self.buffer, self.normalizer, etc.
           return custom_stats
   
   Usage:
       agent = CustomAgent(model, config, device)


5. CUSTOM SAFETY CONSTRAINTS
   
   Inherit from AdaptiveSafety:
   
   class CustomSafety(AdaptiveSafety):
       def validate(self, action_id, temp, speed):
           # Your validation logic
           return is_safe, command
   
   Usage:
       safety = CustomSafety(config)
       is_safe, cmd = safety.validate(action, temp, speed)


# ============================================================================
# PERFORMANCE OPTIMIZATION
# ============================================================================

"""

1. GPU ACCELERATION

   Ensure CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```
   
   Move model to GPU:
   ```python
   model = ActorCriticModel(config).to(device)
   ```
   
   Performance gain: ~5-10x speedup on modern GPUs


2. BATCH SIZE TUNING

   Larger batches → faster updates but less frequent learning
   Smaller batches → slower updates but more frequent learning
   
   Sweet spot: 32-64 for most setups
   
   Modify in config.json:
   ```json
   {
     "learning": {
       "batch_size": 64
     }
   }
   ```


3. SEQUENCE LENGTH OPTIMIZATION

   Longer sequences → better temporal understanding but more memory
   Shorter sequences → less memory but poor temporal coherence
   
   Typical range: 8-32
   
   Modify in config:
   ```python
   config.learning.sequence_length = 16
   ```


4. FEATURE DIMENSION TUNING

   Larger dimensions → more expressive but slower
   Smaller dimensions → faster but less expressive
   
   Balance point: 64-128 for edge devices
   
   Modify in networks.py:
   ```python
   class ActorCriticModel(nn.Module):
       def __init__(self, config):
           self.hidden_dim = 128  # Adjust this
   ```


5. MEMORY OPTIMIZATION

   Pre-allocate buffers:
   ```python
   buffer = ReplayBuffer(capacity=10000, state_shape=(64,))
   ```
   
   Use float32 instead of float64:
   ```python
   tensor = torch.tensor(..., dtype=torch.float32)
   ```
   
   Clear unused tensors:
   ```python
   del intermediate_tensor
   torch.cuda.empty_cache()
   ```


6. PROFILING AND BENCHMARKING

   Profile specific functions:
   ```python
   import cProfile
   cProfile.run('system.train_episode()')
   ```
   
   Time critical sections:
   ```python
   import time
   start = time.perf_counter()
   # code to time
   elapsed = time.perf_counter() - start
   ```


# ============================================================================
# DEBUGGING AND TROUBLESHOOTING
# ============================================================================

"""

1. DEBUGGING TIPS

   Enable verbose logging:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```
   
   Check tensor shapes:
   ```python
   print(f"Input shape: {input_tensor.shape}")
   print(f"Output shape: {output_tensor.shape}")
   ```
   
   Validate gradients:
   ```python
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm={param.grad.norm():.4f}")
   ```
   
   Check for NaN/Inf:
   ```python
   if torch.isnan(loss) or torch.isinf(loss):
       print("Loss contains NaN or Inf!")
   ```


2. COMMON ISSUES AND SOLUTIONS

   Issue: CUDA out of memory
   Solution: Reduce batch_size or sequence_length
   
   Issue: NaN losses during training
   Solution: Reduce learning_rate or check reward computation
   
   Issue: Training not converging
   Solution: Check curriculum_scaling factors or reward normalization
   
   Issue: Safety violations during operation
   Solution: Lower safe_margin or increase failure_penalty
   
   Issue: Slow training speed
   Solution: Check GPU utilization, enable mixed precision (if available)


3. VALIDATION CHECKS

   Before training:
   ```python
   # Check config validity
   config.validate()
   
   # Check hardware connectivity
   printer.get_state()
   
   # Test reward computation
   reward = reward_system.get()
   
   # Verify safety constraints
   is_safe, cmd = safety.validate(0, 210.0, 100.0)
   ```


4. MONITORING TRAINING

   Key metrics to track:
   - Episode return (should increase over time)
   - Policy loss (should decrease)
   - Value loss (should decrease)
   - Safety violations (should decrease)
   - Curriculum stage (should progress smoothly)
   
   Check in training_log.json:
   ```python
   import json
   with open('training_log.json') as f:
       logs = json.load(f)
       returns = [log['episode_return'] for log in logs]
       print(f"Mean return: {np.mean(returns)}")
   ```


# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

"""

1. HYPERPARAMETER TUNING

   Learning rate:
   - Too high: diverges, NaN losses
   - Too low: slow convergence
   - Sweet spot: 1e-4 to 1e-3
   
   Entropy coefficient:
   - Controls exploration vs exploitation
   - Higher: more exploration
   - Lower: more exploitation
   - Typical: 0.001 to 0.01
   
   Value coefficient:
   - Balances actor and critic
   - Higher: more value learning
   - Typical: 0.5
   
   GAE lambda:
   - Trade-off between bias and variance
   - 0.95-0.99: more variance, less bias
   - 0.90-0.95: balanced
   - 0.80-0.90: more bias, less variance


2. CURRICULUM SCHEDULING

   Default 3-stage curriculum:
   Stage 0: reward_scale = 0.5x (exploration)
   Stage 1: reward_scale = 1.0x (refinement)
   Stage 2: reward_scale = 2.0x (production)
   
   Customize in config:
   ```python
   config.learning.curriculum_stages = [0.25, 0.5, 1.0, 2.0, 4.0]
   config.learning.curriculum_steps_per_stage = 500
   ```


3. REWARD SHAPING

   Vision-based reward components:
   - Edge density (0-1): more edges = better quality
   - Blob count (0-1): optimal number of blobs
   - Sharpness (0-1): Laplacian variance
   
   Modify weights in reward_safety.py:
   ```python
   quality = (
       edge_density * 0.7 +  # Adjust weights
       blob_score * 0.3
   )
   ```


# ============================================================================
# CONTRIBUTING AND TESTING
# ============================================================================

"""

1. CODE STYLE

   Follow PEP 8:
   - 4 spaces indentation
   - Max 80 characters per line (exceptions: imports, long strings)
   - Descriptive variable names
   - Docstrings for all public functions
   
   Format with Black:
   ```bash
   black sovereign_v5_final/
   ```
   
   Check with Flake8:
   ```bash
   flake8 sovereign_v5_final/
   ```


2. TESTING GUIDELINES

   Add tests for new features:
   ```python
   def test_your_feature():
       """Test description"""
       # Arrange
       component = YourComponent()
       
       # Act
       result = component.do_something()
       
       # Assert
       assert result == expected_value
   ```
   
   Run tests:
   ```bash
   pytest tests/ -v
   ```
   
   Run with coverage:
   ```bash
   pytest tests/ --cov=. --cov-report=html
   ```


3. DOCUMENTATION

   Every module should have:
   - Docstring explaining purpose
   - Class docstrings with usage examples
   - Parameter descriptions with types
   - Return value descriptions
   - Potential exceptions
   
   Example:
   ```python
   def compute_gae(self, next_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
       \"\"\"
       Compute generalized advantage estimation.
       
       Args:
           next_state: Final state for bootstrapping (shape: (state_dim,))
       
       Returns:
           (advantages, returns): GAE advantages and discounted returns
       
       Raises:
           ValueError: If no transitions stored in buffer
       \"\"\"
   ```


# ============================================================================
# DEPLOYMENT CHECKLIST
# ============================================================================

"""

Before deploying to production:

□ Run full test suite: pytest tests/ -v
□ Code formatting: black sovereign_v5_final/
□ Linting: flake8 sovereign_v5_final/
□ Type checking: mypy sovereign_v5_final/
□ Verify config: python -c "from config import Config; Config().validate()"
□ Test hardware: python -c "from hardware import create_printer; create_printer('mock').get_state()"
□ Test checkpoint save/load: python -c "from agent import SovereignAgent; ..."
□ Review logs: tail -100 sovereign.log
□ Check safety constraints: verify all safety rules are appropriate
□ Backup training data: cp -r checkpoints/ checkpoints.backup/
□ Document configuration: create config documentation
□ Setup monitoring: prepare metrics collection
□ Create runbook: document deployment steps
□ Test recovery: verify checkpoint recovery works
□ Performance test: run on target hardware
□ Security audit: check for hardcoded credentials
□ Documentation: update README if needed


# ============================================================================
# MAINTENANCE AND UPDATES
# ============================================================================

"""

Regular Maintenance Tasks:

Weekly:
- Monitor training progress in training_log.json
- Check for safety violations in logs
- Verify hardware connectivity
- Review error logs in sovereign.log

Monthly:
- Analyze training statistics
- Compare with baseline performance
- Evaluate new configurations
- Plan curriculum adjustments

Quarterly:
- Major version reviews
- Hardware performance analysis
- Safety audit
- Documentation updates

Ongoing:
- Keep PyTorch updated
- Monitor for security patches
- Backup checkpoints regularly
- Archive old training runs


# ============================================================================
# RESOURCES AND LINKS
# ============================================================================

"""

Documentation:
- README.md: User guide and quick start
- BUILD_REPORT.md: Project completion status
- This file: Developer guide

References:
- PPO Paper: https://arxiv.org/abs/1707.06347
- GAE Paper: https://arxiv.org/abs/1506.02438
- PyTorch Docs: https://pytorch.org/docs/stable/index.html

Tools:
- Black: https://black.readthedocs.io/
- Flake8: https://flake8.pycqa.org/
- Pytest: https://docs.pytest.org/
- PyTorch Profiler: https://pytorch.org/docs/stable/profiler.html

Hardware Compatibility:
- OctoPrint: https://octoprint.org/
- Marlin Firmware: https://marlinfw.org/
- PySerial: https://pyserial.readthedocs.io/


# ============================================================================
# CONTACT AND SUPPORT
# ============================================================================

For questions, issues, or feature requests:
1. Check the README.md for common issues
2. Review the troubleshooting section above
3. Check training logs for error details
4. Review test cases for usage examples
5. Consult the examples.py for practical demonstrations

"""
