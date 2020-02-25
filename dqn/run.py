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
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))

    args = parser.parse_args()
    logger.configure()
    set_global_seeds(args.seed)
    Vec = True
    if not Vec:
        env = make_atari(args.env)
        env = bench.Monitor(env, logger.get_dir())
        env = wrap_atari_dqn(env)
    else:
        # venv = DummyVecEnv(np.array([env, _env]))
        venv = SubprocVecEnv([make_atari])
        venv = VecFrameStack(venv, 4)

    policy = partial(CnnPolicy, dueling=args.dueling == 1)

    # model = DQN(
    #     env=env,
    #     policy=policy,
    #     learning_rate=1e-4,
    #     buffer_size=10000,
    #     exploration_fraction=0.1,
    #     exploration_final_eps=0.01,
    #     train_freq=4,
    #     learning_starts=10000,
    #     target_network_update_freq=1000,
    #     gamma=0.99,
    #     prioritized_replay=bool(args.prioritized),
    #     prioritized_replay_alpha=args.prioritized_replay_alpha,
    # )
    model = DQN(
        env=venv,
        policy_class=CnnPolicy,
        learning_rate=1e-4,
        buffer_size=10000,
        double_q=False,
        prioritized_replay=True,
        dueling=True,
        train_freq=4,
        learning_starts=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        target_network_update_freq=1000,
        model_path='atari_test'
    )
    # model.load('atari_Breakout_duel')
    # model.evaluate(num_epsiodes=50)
    model.learn(total_timesteps=args.num_timesteps)
    venv.close()


if __name__ == '__main__':
    main()