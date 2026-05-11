import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from atari_rl.agents import create_agent
from atari_rl.config.settings import Config, ModelType
from atari_rl.utils.atari_wrapper import make_atari_env

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(
            "cuda" if config.device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.log_dir = Path(config.log_dir)

        run_timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.checkpoint_dir = (
            Path(config.checkpoint_dir)
            / config.model_type.value
            / run_timestamp
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        n_channels = config.atari.frame_stack
        self.train_env = make_atari_env(config.atari)
        self.eval_env = make_atari_env(config.atari)
        n_actions = self.train_env.action_space.n

        self.agent = create_agent(config, n_channels, n_actions)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _epsilon(self, step: int) -> float:
        cfg = self.config.exploration
        if cfg.mode.value == "linear":
            ratio = min(step / cfg.epsilon_decay_steps, 1.0)
            return cfg.epsilon_start + (cfg.epsilon_end - cfg.epsilon_start) * ratio
        return cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * (
            cfg.epsilon_decay_rate ** step
        )

    def train(self) -> None:
        config = self.config
        state, _ = self.train_env.reset(seed=config.seed)
        episode_reward = 0.0
        episode_length = 0
        episode = 0
        best_mean_reward = -float("inf")
        start_time = time.time()
        log_time = start_time
        filling = True
        time_limit_reached = False

        total_steps = config.training.total_steps
        total_frames = total_steps * config.atari.frame_skip
        min_size = config.replay.min_size
        time_limit_secs = config.training.time_limit_hours * 3600

        logger.info(
            "Training %s on %s for %d steps (%d frames)  frame_skip=%d  time_limit=%gh",
            config.model_type.value,
            config.atari.env_id,
            total_steps,
            total_frames,
            config.atari.frame_skip,
            config.training.time_limit_hours,
        )

        for step in range(1, total_steps + 1):
            frames = step * config.atari.frame_skip

            if time_limit_secs > 0 and (step % 1000 == 0 or time_limit_reached):
                elapsed = time.time() - start_time
                if elapsed >= time_limit_secs:
                    time_limit_reached = True
                    logger.info(
                        "Time limit of %gh reached at step %d (%d frames, elapsed=%.0fs)",
                        config.training.time_limit_hours,
                        step,
                        frames,
                        elapsed,
                    )
                    self.agent.save(str(self.checkpoint_dir / f"time_limit_step{step}.pt"))
                    break

            epsilon = self._epsilon(step)
            action = self.agent.act(state, epsilon)
            next_state, reward, terminated, truncated, _ = self.train_env.step(action)
            done = terminated or truncated

            reward_clipped = (
                np.clip(reward, -1.0, 1.0) if config.atari.reward_clip else reward
            )
            self.agent.replay.push(
                np.array(state), action, reward_clipped, np.array(next_state), done
            )

            state = next_state
            episode_reward += reward
            episode_length += 1

            if filling and len(self.agent.replay) >= min_size:
                filling = False
                elapsed = time.time() - start_time
                logger.info(
                    "Replay buffer filled (%d samples) in %.0fs",
                    min_size,
                    elapsed,
                )

            if not filling and step % config.training.update_frequency == 0:
                loss = self.agent.update(
                    config.replay.batch_size,
                    config.training.gamma,
                    config.training.tau,
                )
                if loss is not None and step % config.training.log_frequency == 0:
                    self.writer.add_scalar("train/loss", loss, step)
                    self.writer.add_scalar("train/epsilon", epsilon, step)

            if step % config.training.target_update_frequency == 0:
                self.agent.soft_update_target(config.training.tau)

            if done:
                episode += 1
                self.writer.add_scalar(
                    "train/episode_reward", episode_reward, step
                )
                self.writer.add_scalar(
                    "train/episode_length", episode_length, step
                )
                state, _ = self.train_env.reset()
                episode_reward = 0.0
                episode_length = 0

            if step % config.training.eval_frequency == 0:
                mean_reward = self._evaluate()
                self.writer.add_scalar("eval/mean_reward", mean_reward, step)
                self.writer.flush()

                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                fps = frames / elapsed
                eta = (total_steps - step) / steps_per_sec
                logger.info(
                    "Step %d/%d  frames=%d  eps=%.3f  eval_reward=%.1f  %.0f steps/s  %.0f fps  ETA=%.0fmin",
                    step,
                    total_steps,
                    frames,
                    epsilon,
                    mean_reward,
                    steps_per_sec,
                    fps,
                    eta / 60,
                )

                if mean_reward > best_mean_reward:
                    best_mean_reward = mean_reward
                    self.agent.save(str(self.checkpoint_dir / f"best_{step}.pt"))

            if step % config.training.save_frequency == 0:
                self.agent.save(
                    str(self.checkpoint_dir / f"step{step}.pt")
                )

            if time.time() - log_time >= 30:
                log_time = time.time()
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed
                fps = frames / elapsed
                eta = (total_steps - step) / steps_per_sec
                phase = "filling buffer" if filling else "training"
                logger.info(
                    "Step %d/%d  frames=%d  eps=%.3f  buffer=%d  %s  %.0f steps/s  %.0f fps  ETA=%.0fmin",
                    step,
                    total_steps,
                    frames,
                    epsilon,
                    len(self.agent.replay),
                    phase,
                    steps_per_sec,
                    fps,
                    eta / 60,
                )

        total_elapsed = time.time() - start_time
        final_step = step if time_limit_reached else total_steps
        final_frames = final_step * config.atari.frame_skip
        logger.info(
            "Training complete  model=%s  env=%s  steps=%d  frames=%d  episodes=%d  "
            "time=%.0fs (%.2fh)  %.0f steps/s  %.0f fps  best_reward=%.1f",
            config.model_type.value,
            config.atari.env_id,
            final_step,
            final_frames,
            episode,
            total_elapsed,
            total_elapsed / 3600,
            final_step / total_elapsed,
            final_frames / total_elapsed,
            best_mean_reward if best_mean_reward != -float("inf") else 0.0,
        )
        self.train_env.close()
        self.eval_env.close()
        self.writer.close()

    def test(self, checkpoint_path: str | None = None, render: bool = False, stopRepeated: int = 20) -> list[float]:
        if checkpoint_path:
            ckpt_meta = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            ).get("metadata", {})

            ckpt_env_id = ckpt_meta.get("env_id")
            if ckpt_env_id and ckpt_env_id != self.config.atari.env_id:
                logger.info(
                    "Using env_id=%s from checkpoint (config had %s)",
                    ckpt_env_id, self.config.atari.env_id,
                )
                self.config.atari.env_id = ckpt_env_id

            ckpt_model_type = ckpt_meta.get("model_type")
            if ckpt_model_type and ckpt_model_type != self.config.model_type.value:
                logger.info(
                    "Using model_type=%s from checkpoint (config had %s)",
                    ckpt_model_type, self.config.model_type.value,
                )
                self.config.model_type = ModelType(ckpt_model_type)

        render_mode = "human" if render else None
        env = make_atari_env(self.config.atari, render_mode=render_mode)

        if checkpoint_path:
            n_channels = self.config.atari.frame_stack
            n_actions = env.action_space.n
            self.agent = create_agent(self.config, n_channels, n_actions)
            self.agent.load(checkpoint_path)

        self.agent.online.eval()
        rewards = []
        n_episodes = self.config.evaluation.episodes

        logger.info(
            "Testing %s on %s for %d episodes (render=%s)",
            self.config.model_type.value,
            self.config.atari.env_id,
            n_episodes,
            render,
        )

        for ep in range(1, n_episodes + 1):
            state, _ = env.reset()
            total_reward = 0.0
            done = False
            steps = 0
            repeatedSteps = 0
            previousAction = None

            while not done:
                action = self.agent.act(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = env.step(action)

                if previousAction == action:
                    if repeatedSteps >= stopRepeated:
                        terminated = True
                    else:
                        repeatedSteps += 1
                else:
                    previousAction = action
                    repeatedSteps = 0

                total_reward += reward
                done = terminated or truncated
                steps += 1

            rewards.append(total_reward)
            logger.info(
                "Episode %d/%d  reward=%.0f  steps=%d",
                ep,
                n_episodes,
                total_reward,
                steps,
            )

        mean = float(np.mean(rewards))
        std = float(np.std(rewards))
        logger.info(
            "Test complete  mean_reward=%.1f  std=%.1f  min=%.0f  max=%.0f",
            mean,
            std,
            min(rewards),
            max(rewards),
        )

        env.close()
        return rewards

    def _evaluate(self) -> float:
        self.agent.online.eval()
        rewards = []

        for _ in range(self.config.training.eval_episodes):
            state, _ = self.eval_env.reset()
            total = 0.0
            done = False

            while not done:
                action = self.agent.act(state, epsilon=0.0)
                state, reward, terminated, truncated, _ = self.eval_env.step(action)
                total += reward
                done = terminated or truncated

            rewards.append(total)

        self.agent.online.train()
        return float(np.mean(rewards))
