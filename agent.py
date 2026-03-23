"""
Sovereign Agent - Core RL Agent with PPO Training
Production-grade implementation with all fixes applied
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import deque
import hashlib

from networks import ActorCriticModel, Perception
from config import Config

logger = logging.getLogger(__name__)


class RunningNormalizer:
    """
    Bias-corrected running mean/variance normalizer
    Uses Welford's online algorithm for numerical stability
    """
    
    def __init__(self, shape: int):
        self.shape = shape
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, batch: np.ndarray):
        """
        Update statistics with new batch
        Args:
            batch: (batch_size, shape) numpy array
        """
        batch = batch.reshape(-1, self.shape)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        # Welford's online update
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean += delta * batch_count / total_count

        # Parallel algorithm for variance
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize input using running statistics"""
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


class ReplayBuffer:
    """
    Efficient memory-pinned replay buffer using numpy arrays
    Prevents fragmentation and improves GPU transfer speed
    """
    
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity = capacity
        self.state_dim = state_dim
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Pre-allocate numpy arrays
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros(capacity, dtype=np.int64)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.log_prob = np.zeros(capacity, dtype=np.float32)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)

    def add(self, state, action, reward, log_prob, value, done):
        """Add single transition"""
        idx = self.ptr
        self.state[idx] = state
        self.action[idx] = action
        self.reward[idx] = reward
        self.log_prob[idx] = log_prob
        self.value[idx] = value
        self.done[idx] = done
        
        self.ptr = (idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get_all(self) -> Dict[str, torch.Tensor]:
        """Get all stored transitions as tensors"""
        idx = slice(0, self.size)
        return {
            'state': torch.FloatTensor(self.state[idx]).to(self.device),
            'action': torch.LongTensor(self.action[idx]).to(self.device),
            'reward': torch.FloatTensor(self.reward[idx]).to(self.device),
            'log_prob': torch.FloatTensor(self.log_prob[idx]).to(self.device),
            'value': torch.FloatTensor(self.value[idx]).to(self.device),
            'done': torch.BoolTensor(self.done[idx]).to(self.device),
        }

    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0


class SovereignAgent:
    """
    Complete autonomous printer AI agent
    Implements PPO with sequence-batched training
    """
    
    def __init__(self, config: Config, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models
        self.model = ActorCriticModel(
            sensor_dim=8,
            latent_dim=128,
            action_dim=4,
            device=self.device
        ).to(self.device)
        
        # Optimizer with separate learning rates
        self.optimizer = torch.optim.Adam([
            {'params': self.model.perception.parameters(), 
             'lr': config.learning.lr_perception},
            {'params': self.model.policy.parameters(), 
             'lr': config.learning.lr}
        ])
        
        # Normalizer and buffers
        self.normalizer = RunningNormalizer(8)
        self.replay_buffer = ReplayBuffer(2048, 8, self.device)
        
        # Tracking
        self.step_counter = 0
        self.episode_counter = 0
        self.curriculum_stage = 0
        self.session_id = hashlib.md5(str(np.random.rand()).encode()).hexdigest()[:10]
        
        logger.info(f"SovereignAgent initialized on {self.device}")
        logger.info(f"Session ID: {self.session_id}")

    def act(
        self,
        sensor_raw: np.ndarray,
        vision_frame: Optional[torch.Tensor] = None,
        reset_hidden: bool = False
    ) -> Tuple[int, float, float, np.ndarray]:
        """
        Select action and return auxiliary info
        Args:
            sensor_raw: raw sensor readings
            vision_frame: camera frame (optional)
            reset_hidden: whether to reset LSTM hidden state
        Returns:
            (action, log_prob, value, normalized_state)
        """
        # Update normalizer and normalize state
        self.normalizer.update(sensor_raw.reshape(1, -1))
        sensor_norm = self.normalizer.normalize(sensor_raw)
        
        s_t = torch.FloatTensor(sensor_norm).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs, value = self.model(s_t, vision_frame, reset_hidden=reset_hidden)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return (
            action.item(),
            log_prob.item(),
            value.item(),
            sensor_norm
        )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Store transition in replay buffer"""
        self.replay_buffer.add(state, action, reward, log_prob, value, done)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE)
        CRITICAL FIX: Proper tensor operations to avoid precision loss
        """
        device = rewards.device
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=device)
        next_value = torch.tensor(0.0, device=device)
        
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t].float()
            delta = rewards[t] + self.config.learning.gamma * next_value * non_terminal - values[t]
            advantages[t] = gae = (
                delta + 
                self.config.learning.gamma * self.config.learning.gae_lambda * non_terminal * gae
            )
            next_value = values[t]
        
        returns = advantages + values
        return advantages, returns

    def train_ppo_step(self):
        """
        Sequence-batched PPO update
        CRITICAL FIX: No shuffling to preserve temporal order for LSTM
        """
        if self.replay_buffer.size < self.config.learning.batch_size:
            return
        
        # Get batch
        batch = self.replay_buffer.get_all()
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        old_log_probs = batch['log_prob']
        values = batch['value']
        dones = batch['done']
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Normalize advantages (CRITICAL for stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO updates (no shuffling!)
        for epoch in range(self.config.learning.epochs_per_update):
            # Forward pass
            self.model.reset_policy_hidden(states.size(0), self.device)
            probs, state_values = self.model(states, None, reset_hidden=False)
            
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.learning.clip_ratio,
                1.0 + self.config.learning.clip_ratio
            ) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            loss = (
                policy_loss +
                self.config.learning.value_coef * value_loss -
                self.config.learning.entropy_coef * entropy
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Log gradient norm (for debugging)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            logger.debug(f"Epoch {epoch}: Loss={loss.item():.4f}, GradNorm={total_norm:.4f}")
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.learning.max_grad_norm
            )
            self.optimizer.step()
            
            # Detach hidden state to prevent graph leaks
            self.model.detach_policy_hidden()
        
        self.replay_buffer.clear()

    def save_checkpoint(self, path: Optional[str] = None):
        """Save complete model checkpoint"""
        if path is None:
            path = Path(self.config.checkpoint_dir) / f"sovereign_step{self.step_counter}.pt"
        
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'normalizer_mean': self.normalizer.mean,
            'normalizer_var': self.normalizer.var,
            'step_counter': self.step_counter,
            'episode_counter': self.episode_counter,
            'curriculum_stage': self.curriculum_stage,
            'session_id': self.session_id,
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.normalizer.mean = checkpoint['normalizer_mean']
        self.normalizer.var = checkpoint['normalizer_var']
        self.step_counter = checkpoint['step_counter']
        self.episode_counter = checkpoint['episode_counter']
        self.curriculum_stage = checkpoint['curriculum_stage']
        self.session_id = checkpoint['session_id']
        logger.info(f"Checkpoint loaded from {path}")
