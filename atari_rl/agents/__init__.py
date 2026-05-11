from atari_rl.agents.dqn_agent import DQNAgent
from atari_rl.agents.nec_agent import NECAgent
from atari_rl.config.settings import ModelType

_AGENT_REGISTRY = {
    ModelType.DQN: DQNAgent,
    ModelType.DOUBLE_DQN: DQNAgent,
    ModelType.DUELING_DQN: DQNAgent,
    ModelType.NEC: NECAgent,
}


def create_agent(config, n_channels: int, n_actions: int):
    agent_cls = _AGENT_REGISTRY[config.model_type]
    return agent_cls(config, n_channels, n_actions)


__all__ = ["DQNAgent", "NECAgent", "create_agent"]
