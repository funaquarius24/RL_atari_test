import argparse
import logging
import random
import sys
 
import numpy as np
import torch

from atari_rl.config.settings import Config, ModelType
from atari_rl.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atari deep RL — DQN / Double DQN / Dueling DQN / NEC"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Run mode (default: train)",
    )
    parser.add_argument(
        "--env", type=str, default="PongNoFrameskip-v4", help="Atari env ID"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dqn",
        choices=["dqn", "double_dqn", "dueling_dqn", "nec"],
        help="Model architecture",
    )
    parser.add_argument(
        "--total-steps", type=int, default=1_000_000, help="Total training steps"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Replay batch size")
    parser.add_argument(
        "--buffer-capacity", type=int, default=100_000, help="Replay buffer capacity"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cuda/cpu)")
    parser.add_argument(
        "--log-dir", type=str, default="runs", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--time-limit", type=float, default=0.0, help="Max training time in hours (0 = no limit)"
    )

    # Test mode arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for test mode",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during test mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for test mode",
    )
    return parser


def _prepare_config(args) -> Config:
    config = Config(
        seed=args.seed,
        device=args.device,
        model_type=ModelType(args.model),
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    config.atari.env_id = args.env
    config.training.total_steps = args.total_steps
    config.training.learning_rate = args.lr
    config.replay.batch_size = args.batch_size
    config.replay.capacity = args.buffer_capacity
    config.training.time_limit_hours = args.time_limit
    config.evaluation.episodes = args.episodes
    config.evaluation.render = args.render
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _prepare_config(args)
    _set_seed(config.seed)

    trainer = Trainer(config)

    print(args)

    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":

        trainer.test(checkpoint_path=args.checkpoint, render=args.render)


if __name__ == "__main__":
    main()
