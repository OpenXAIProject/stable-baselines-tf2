from dqn.dqn import DQN, ReplayBuffer
from dqn.policy import MlpPolicy


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN
    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)