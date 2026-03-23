"""
Sovereign-v5.0: Autonomous Edge AI for 3D Printers

A production-ready reinforcement learning system using PPO, LSTM, and federated learning
for autonomous control of 3D printers.

Key Components:
- config: Configuration management
- networks: Neural network architectures (Vision CNN, LSTM Actor-Critic)
- agent: RL agent with PPO training and GAE
- reward_safety: Vision-based rewards and adaptive safety layer
- hardware: Printer interfaces (Mock, OctoPrint, Serial)
- federated: Federated learning coordination
- main: Main training loop orchestrator

Example Usage:
    from main import SovereignAutonomousSystem
    
    system = SovereignAutonomousSystem(printer_mode='mock')
    system.train(num_episodes=1000)
"""

__version__ = '5.0.0'
__author__ = 'Autonomous Systems Lab'
__all__ = [
    'config',
    'networks',
    'agent',
    'reward_safety',
    'hardware',
    'federated',
    'main',
]

import logging

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
