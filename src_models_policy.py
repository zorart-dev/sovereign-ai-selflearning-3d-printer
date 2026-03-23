"""Actor-Critic policy networks."""

import torch
import torch.nn as nn


class LSTMActorCritic(nn.Module):
    """LSTM-based actor-critic for temporal reasoning."""
    
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, action_dim: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, action_dim), nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.hidden = None
    
    def forward(self, latent: torch.Tensor, reset_hidden: bool = False):
        batch_size = latent.size(0)
        device = latent.device
        
        if reset_hidden or self.hidden is None:
            self.hidden = (
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
                torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
            )
        
        out, self.hidden = self.lstm(latent.unsqueeze(1), self.hidden)
        out = out.squeeze(1)
        probs = self.actor(out)
        value = self.critic(out)
        return probs, value
    
    def detach_hidden(self):
        if self.hidden is not None:
            self.hidden = tuple(h.detach() for h in self.hidden)