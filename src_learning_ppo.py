"""PPO training algorithm."""

import torch
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Handles PPO update logic."""
    
    def __init__(self, perception, policy, optimizer, config):
        self.perception = perception
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.device = next(perception.parameters()).device
    
    def update(self, buffer):
        """Perform single PPO update."""
        if len(buffer) < self.config.batch_size:
            return
        
        states = torch.FloatTensor(np.array([t['state'] for t in buffer])).to(self.device)
        actions = torch.LongTensor([t['action'] for t in buffer]).to(self.device)
        rewards = np.array([t['reward'] for t in buffer])
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in buffer]).to(self.device)
        values = torch.FloatTensor([t['value'] for t in buffer]).to(self.device)
        dones = np.array([t['done'] for t in buffer])
        
        # GAE
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * (values[t+1].item() if t+1 < len(values) else 0) * non_terminal - values[t].item()
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * non_terminal * last_gae
        
        returns = advantages + values.cpu().numpy()
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.config.epochs_per_update):
            latent = self.perception(states)
            self.policy.hidden = None
            probs, state_values = self.policy(latent)
            
            from torch.distributions import Categorical
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.perception.parameters()) + list(self.policy.parameters()),
                self.config.max_grad_norm
            )
            self.optimizer.step()
            self.policy.detach_hidden()