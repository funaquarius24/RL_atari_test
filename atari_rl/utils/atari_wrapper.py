import logging

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

from atari_rl.config.settings import AtariConfig

logger = logging.getLogger(__name__)


def make_atari_env(config: AtariConfig, render_mode: str | None = None) -> gym.Env:
    try:
        import ale_py

        gym.register_envs(ale_py)
    except ImportError:
        logger.error(
            "ale_py is not installed. Run: pip install gymnasium[atari]"
        )
        raise

    env = gym.make(config.env_id, render_mode=render_mode)
    env = AtariPreprocessing(
        env,
        screen_size=config.screen_size,
        grayscale_obs=config.grayscale,
        frame_skip=config.frame_skip,
        noop_max=config.noop_max,
    )
    env = FrameStackObservation(env, stack_size=config.frame_stack)
    return env
