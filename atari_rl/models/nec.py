import torch
import torch.nn as nn


class NECModel(nn.Module):
    def __init__(self, n_channels: int, n_actions: int, embedding_dim: int = 512):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, embedding_dim),
            nn.LeakyReLU(0.01),
        )
        self.advantage = nn.Linear(embedding_dim, n_actions)
        self.value = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.features(x))
        adv = self.advantage(h)
        val = self.value(h)
        return val + adv - adv.mean(dim=-1, keepdim=True)

    def forward_with_embedding(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(self.features(x))
        adv = self.advantage(h)
        val = self.value(h)
        q = val + adv - adv.mean(dim=-1, keepdim=True)
        return q, h

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.features(x))
