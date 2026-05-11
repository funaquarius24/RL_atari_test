import logging
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class DND:
    def __init__(self, key_dim: int, capacity: int, lr: float = 0.1, knn_k: int = 50):
        self.key_dim = key_dim
        self.capacity = capacity
        self.lr = lr
        self.knn_k = knn_k

        self.keys = np.empty((0, key_dim), dtype=np.float32)
        self.values = np.empty(0, dtype=np.float32)
        self.index = None
        self._dirty = False

    def _rebuild_index(self):
        if len(self.keys) == 0:
            self.index = None
            self._dirty = False
            return
        norms = np.linalg.norm(self.keys, axis=1, keepdims=True)
        normalized = self.keys / np.maximum(norms, 1e-10)
        index = faiss.IndexFlatIP(self.key_dim)
        index.add(normalized.astype(np.float32))
        self.index = index
        self._dirty = False

    def lookup(self, key: np.ndarray) -> tuple[float, float]:
        if len(self.keys) == 0:
            return 0.0, 0.0
        if self._dirty or self.index is None:
            self._rebuild_index()

        key_norm = key / (np.linalg.norm(key) + 1e-10)
        k = min(self.knn_k, len(self.keys))
        distances, indices = self.index.search(
            key_norm.reshape(1, -1).astype(np.float32), k
        )

        temp = 0.1
        weights = np.exp((distances[0] - 1.0) / temp)
        weights /= weights.sum() + 1e-10

        q = float(np.dot(weights, self.values[indices[0]]))
        return q, k / self.knn_k

    def insert(self, key: np.ndarray, value: float) -> None:
        key_norm = key / (np.linalg.norm(key) + 1e-10)

        if len(self.keys) > 0:
            if self._dirty or self.index is None:
                self._rebuild_index()
            distances, indices = self.index.search(
                key_norm.reshape(1, -1).astype(np.float32), 1
            )
            if distances[0, 0] > 0.99:
                idx = indices[0, 0]
                self.values[idx] += self.lr * (value - self.values[idx])
                return

        if len(self.keys) >= self.capacity:
            keep = max(1, int(self.capacity * 0.75))
            self.keys = self.keys[-keep:]
            self.values = self.values[-keep:]
            self._dirty = True

        self.keys = np.append(self.keys, key_norm.reshape(1, -1), axis=0)
        self.values = np.append(self.values, value)
        self._dirty = True

    def __len__(self) -> int:
        return len(self.keys)

    def state_dict(self) -> dict:
        return {"keys": self.keys, "values": self.values}

    def load_state_dict(self, state: dict) -> None:
        self.keys = state["keys"]
        self.values = state["values"]
        self._dirty = True
