"""
Neural Network Components for Sovereign-v5.0
Includes fixes for LSTM hidden state management and vision encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class TinyVisionCNN(nn.Module):
    """
    Lightweight vision encoder inspired by MobileNetV3
    Input: (batch, 3, 64, 64) RGB frames
    Output: (batch, out_dim) feature vector
    Memory: ~2 MB
    """
    
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: (batch, 3, H, W) - RGB image
        Returns:
            (batch, out_dim) - feature vector
        """
        # Ensure input is in [0, 1]
        if x.max() > 1.0:
            x = x / 255.0
        
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Perception(nn.Module):
    """
    Multi-modal perception encoder
    Fuses sensor data with vision information
    """
    
    def __init__(
        self,
        sensor_dim: int = 8,
        vision_dim: int = 64,
        fused_dim: int = 128,
        device: torch.device = None
    ):
        super().__init__()
        self.sensor_dim = sensor_dim
        self.device = device or torch.device('cpu')
        
        self.vision_encoder = TinyVisionCNN(out_dim=vision_dim)
        self.sensor_fc = nn.Linear(sensor_dim, fused_dim // 2)
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + fused_dim // 2, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )

    def forward(
        self,
        sensors: torch.Tensor,
        vision: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass
        Args:
            sensors: (batch, sensor_dim) - normalized sensor readings
            vision: (batch, 3, 64, 64) - RGB camera frames [0, 255] or [0, 1]
        Returns:
            (batch, fused_dim) - fused representation
        """
        s = self.sensor_fc(sensors)
        
        if vision is not None:
            v = self.vision_encoder(vision)
            fused = torch.cat([s, v], dim=1)
        else:
            # Use zero padding if vision unavailable
            v_dim = self.vision_encoder.fc.out_features
            v = torch.zeros(s.size(0), v_dim, device=sensors.device)
            fused = torch.cat([s, v], dim=1)
        
        return self.fusion(fused)


class LSTMActorCritic(nn.Module):
    """
    LSTM-based Actor-Critic network
    Maintains temporal memory for print state sequences
    
    CRITICAL FIX: Proper hidden state shape management
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 64,
        action_dim: int = 4,
        device: torch.device = None
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.device = device or torch.device('cpu')
        
        # LSTM: takes latent representations
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Actor: outputs action probabilities
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic: outputs state value
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Hidden state (h, c)
        self._hidden = None

    @property
    def hidden(self):
        return self._hidden

    @hidden.setter
    def hidden(self, value):
        self._hidden = value

    def reset_hidden(self, batch_size: int, device: torch.device = None):
        """
        Reset hidden state with correct dimensions
        CRITICAL FIX: (num_layers, batch_size, hidden_dim)
        """
        device = device or self.device
        self._hidden = (
            torch.zeros(1, batch_size, self.hidden_dim, device=device),
            torch.zeros(1, batch_size, self.hidden_dim, device=device)
        )

    def detach_hidden(self):
        """
        Detach hidden state from computation graph
        Prevents memory leaks and gradient explosions
        """
        if self._hidden is not None:
            self._hidden = tuple(h.detach() for h in self._hidden)

    def forward(
        self,
        latent: torch.Tensor,
        reset_hidden: bool = False
    ) -> tuple:
        """
        Forward pass
        Args:
            latent: (batch, latent_dim) - encoded state
            reset_hidden: bool - whether to reset hidden state
        Returns:
            (probs, value) - action probabilities and state value
        """
        batch_size = latent.size(0)
        device = latent.device
        
        # Initialize or reset hidden state
        if reset_hidden or self._hidden is None:
            self.reset_hidden(batch_size, device)
        
        # Add sequence dimension: (batch, latent_dim) -> (batch, 1, latent_dim)
        latent_seq = latent.unsqueeze(1)
        
        # LSTM forward
        lstm_out, self._hidden = self.lstm(latent_seq, self._hidden)
        
        # Remove sequence dimension: (batch, 1, hidden_dim) -> (batch, hidden_dim)
        lstm_out = lstm_out.squeeze(1)
        
        # Actor and critic heads
        probs = self.actor(lstm_out)
        value = self.critic(lstm_out)
        
        return probs, value


class ActorCriticModel(nn.Module):
    """
    Complete Actor-Critic model combining perception and LSTM policy
    """
    
    def __init__(
        self,
        sensor_dim: int = 8,
        latent_dim: int = 128,
        action_dim: int = 4,
        device: torch.device = None
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        
        self.perception = Perception(
            sensor_dim=sensor_dim,
            fused_dim=latent_dim,
            device=self.device
        )
        
        self.policy = LSTMActorCritic(
            latent_dim=latent_dim,
            action_dim=action_dim,
            device=self.device
        )

    def forward(
        self,
        sensors: torch.Tensor,
        vision: torch.Tensor = None,
        reset_hidden: bool = False
    ) -> tuple:
        """
        Complete forward pass
        Args:
            sensors: (batch, sensor_dim) - normalized sensors
            vision: (batch, 3, 64, 64) - camera frames
            reset_hidden: bool - reset LSTM hidden state
        Returns:
            (probs, value) - action probabilities and value
        """
        latent = self.perception(sensors, vision)
        probs, value = self.policy(latent, reset_hidden=reset_hidden)
        return probs, value

    def detach_policy_hidden(self):
        """Detach policy hidden state"""
        self.policy.detach_hidden()

    def reset_policy_hidden(self, batch_size: int, device: torch.device = None):
        """Reset policy hidden state"""
        self.policy.reset_hidden(batch_size, device or self.device)
