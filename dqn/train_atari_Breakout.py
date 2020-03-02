import numpy as np

import argparse
from functools import partial

import bench, logger
from common.misc_util import set_global_seeds
from common.atari_wrappers import make_atari
from dqn import DQN
# from dqn import wrap_atari_dqn
from policy import CnnPolicy, MlpPolicy

from common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize, VecEnvWrapper, VecVideoRecorder


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN
    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from common.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)


def main():
    """
    Run the atari test
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--prioritized', type=int, default=1)
    parser.add_argument('--dueling', type=int, default=1)
    parser.add_argument('--prioritized-replay-alpha', type=float, default=0.6)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)

    env = make_atari(args.env)
    env.action_space.seed(args.seed)
    env = bench.Monitor(env, logger.get_dir())
    env = wrap_atari_dqn(env)

    model = DQN(
        env=env,
        policy_class=CnnPolicy,
        buffer_size=10000,
        learning_rate=1e-4,
        learning_starts=10000,
        target_network_update_freq=1000,
        train_freq=4,
        exploration_final_eps=0.01,
        exploration_fraction=0.1,
        prioritized_replay=True,
        model_path='atari_test_Breakout'
    )
    model.learn(total_timesteps=args.num_timesteps)
    env.close()


if __name__ == '__main__':
    main()