import logging

import torch
import torch.nn.functional as F
from torch import nn, optim

from atari_rl.config.settings import Config, ModelType
from atari_rl.models import DQN, DoubleDQN, DuelingDQN
from atari_rl.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)

_MODEL_REGISTRY = {
    ModelType.DQN: DQN,
    ModelType.DOUBLE_DQN: DoubleDQN,
    ModelType.DUELING_DQN: DuelingDQN,
}


class DQNAgent:
    def __init__(self, config: Config, n_channels: int, n_actions: int):
        self.config = config
        self.device = _resolve_device(config.device)
        self.n_actions = n_actions

        model_cls = _MODEL_REGISTRY[config.model_type]
        self.online = model_cls(n_channels, n_actions).to(self.device)
        self.target = model_cls(n_channels, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(), lr=config.training.learning_rate
        )
        self.replay = ReplayBuffer(config.replay.capacity)
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def act(self, state, epsilon: float = 0.0) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_t)
        return q_values.argmax(dim=1).item()

    def update(self, batch_size: int, gamma: float, tau: float) -> float | None:
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_current = self.online(states).gather(1, actions)

        with torch.no_grad():
            if self.config.model_type == ModelType.DOUBLE_DQN:
                best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                q_next = self.target(next_states).gather(1, best_actions)
            else:
                q_next = self.target(next_states).max(dim=1, keepdim=True).values

        q_target = rewards + gamma * q_next * (1 - dones)
        loss = F.smooth_l1_loss(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()

        self._step += 1
        return loss.item()

    def soft_update_target(self, tau: float) -> None:
        for tp, op in zip(self.target.parameters(), self.online.parameters()):
            tp.data.copy_(tau * op.data + (1 - tau) * tp.data)

    def hard_update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str) -> None:
        n_params = sum(p.numel() for p in self.online.parameters())
        metadata = {
            "model_type": self.config.model_type.value,
            "env_id": self.config.atari.env_id,
            "n_channels": self.online.features[0].in_channels,
            "n_actions": self.n_actions,
            "n_params": n_params,
            "learning_rate": self.config.training.learning_rate,
            "gamma": self.config.training.gamma,
            "tau": self.config.training.tau,
            "device": str(self.device),
            "date_time": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "framework": f"torch {__import__('torch').__version__}",
        }
        torch.save(
            {
                "metadata": metadata,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._step,
            },
            path,
        )
        logger.info(
            "Checkpoint saved to %s  model=%s  step=%d  params=%d  %s",
            path,
            metadata["model_type"],
            self._step,
            n_params,
            metadata["date_time"],
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get("metadata")
        if metadata:
            logger.info(
                "Checkpoint:  model=%s  env=%s  step=%d  params=%d  %s",
                metadata.get("model_type", "?"),
                metadata.get("env_id", "?"),
                checkpoint["step"],
                metadata.get("n_params", "?"),
                metadata.get("date_time", "?"),
            )
        self.online.load_state_dict(checkpoint["online"])
        self.target.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._step = checkpoint["step"]


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
