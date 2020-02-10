import argparse

import gym

from dqn import DQN
from policy import MlpPolicy, CnnPolicy


def main(args):
    """
    Run a trained model for the cartpole problem
    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("CartPole-v0")
    model = DQN(
        env=env,
        policy_class=MlpPolicy,
        learning_rate=5e-4,
        buffer_size=50000,
        double_q=False,
        prioritized_replay=True,
        dueling=True,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        model_path='cartpole_model'
    )
    model = model.load("cartpole_model")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        # No render is only used for automatic testing
        if args.no_render:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on cartpole")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    args = parser.parse_args()
    main(args)