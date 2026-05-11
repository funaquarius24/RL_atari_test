from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class ModelType(str, Enum):
    DQN = "dqn"
    DOUBLE_DQN = "double_dqn"
    DUELING_DQN = "dueling_dqn"
    NEC = "nec"


class ExplorationMode(str, Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class AtariConfig:
    env_id: str = "PongNoFrameskip-v4"
    frame_stack: int = 4
    frame_skip: int = 4
    screen_size: int = 84
    grayscale: bool = True
    noop_max: int = 30
    fire_on_reset: bool = True
    reward_clip: bool = True


@dataclass
class ReplayBufferConfig:
    capacity: int = 100_000
    min_size: int = 10_000
    batch_size: int = 32


@dataclass
class TrainingConfig:
    total_steps: int = 1_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    tau: float = 0.005
    update_frequency: int = 4
    target_update_frequency: int = 1_000
    max_grad_norm: float = 10.0
    eval_episodes: int = 5
    eval_frequency: int = 50_000
    log_frequency: int = 1_000
    save_frequency: int = 100_000
    time_limit_hours: float = 0.0


@dataclass
class EvaluationConfig:
    episodes: int = 10
    render: bool = False


@dataclass
class ExplorationConfig:
    mode: ExplorationMode = ExplorationMode.LINEAR
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 250_000
    epsilon_decay_rate: float = 0.999


@dataclass
class DNDConfig:
    capacity: int = 50_000
    lr: float = 0.1
    knn_k: int = 50


@dataclass
class Config:
    seed: int = 42
    device: str = "auto"
    model_type: ModelType = ModelType.DQN
    atari: AtariConfig = field(default_factory=AtariConfig)
    replay: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    dnd: DNDConfig = field(default_factory=DNDConfig)
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
