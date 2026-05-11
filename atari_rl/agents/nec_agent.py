import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from atari_rl.config.settings import Config
from atari_rl.memory.dnd import DND
from atari_rl.models.nec import NECModel
from atari_rl.utils.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class NECAgent:
    def __init__(self, config: Config, n_channels: int, n_actions: int):
        self.config = config
        self.device = _resolve_device(config.device)
        self.n_actions = n_actions

        self.online = NECModel(n_channels, n_actions).to(self.device)
        self.target = NECModel(n_channels, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(
            self.online.parameters(), lr=config.training.learning_rate
        )
        self.replay = ReplayBuffer(config.replay.capacity)

        self.dnds = [
            DND(
                key_dim=self.online.embedding_dim,
                capacity=config.dnd.capacity,
                lr=config.dnd.lr,
                knn_k=config.dnd.knn_k,
            )
            for _ in range(n_actions)
        ]

        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def _get_q_values(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_net = self.online(state_t).squeeze(0).cpu().numpy()
            h_target = self.target.encode(state_t).squeeze(0).cpu().numpy()

        q_dnd = np.zeros(self.n_actions, dtype=np.float32)
        conf = np.zeros(self.n_actions, dtype=np.float32)
        for a in range(self.n_actions):
            q_dnd[a], conf[a] = self.dnds[a].lookup(h_target)

        alpha = np.clip(conf, 0.0, 1.0)
        return alpha * q_dnd + (1.0 - alpha) * q_net

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.n_actions, (1,)).item()
        q_values = self._get_q_values(state)
        return int(q_values.argmax())

    def update(self, batch_size: int, gamma: float, tau: float) -> float | None:
        if len(self.replay) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones_t = torch.tensor(dones, device=self.device).unsqueeze(1)

        q_current = self.online(states_t)
        q_pred = q_current.gather(1, actions_t)

        with torch.no_grad():
            h_next = self.target.encode(next_states_t).cpu().numpy()

            q_target = rewards_t.clone()
            for i in range(batch_size):
                if not bool(dones[i]):
                    best_q = -float("inf")
                    h_i = h_next[i]
                    for a in range(self.n_actions):
                        dnd_q, _ = self.dnds[a].lookup(h_i)
                        if dnd_q > best_q:
                            best_q = dnd_q
                    if best_q != -float("inf"):
                        q_target[i] += gamma * best_q

        loss = F.mse_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.online.parameters(), self.config.training.max_grad_norm
        )
        self.optimizer.step()

        with torch.no_grad():
            h_current = self.target.encode(states_t).cpu().numpy()
            q_target_np = q_target.squeeze(1).cpu().numpy()
            for i in range(batch_size):
                self.dnds[actions[i]].insert(h_current[i], float(q_target_np[i]))

        self._step += 1
        return loss.item()

    def soft_update_target(self, tau: float) -> None:
        self.target.load_state_dict(self.online.state_dict())

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
            "date_time": datetime.now().isoformat(timespec="seconds"),
            "framework": f"torch {torch.__version__}",
        }
        torch.save(
            {
                "metadata": metadata,
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self._step,
                "dnds": [dnd.state_dict() for dnd in self.dnds],
            },
            path,
        )
        logger.info(
            "Checkpoint saved to %s  model=%s  step=%d  params=%d  dnd_entries=%s  %s",
            path,
            metadata["model_type"],
            self._step,
            n_params,
            [len(d) for d in self.dnds],
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
        if "dnds" in checkpoint:
            for dnd, state in zip(self.dnds, checkpoint["dnds"]):
                dnd.load_state_dict(state)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
