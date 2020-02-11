import argparse

import gym

from dqn import DQN
from policy import MlpPolicy, CnnPolicy

def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("Pong-v0")

    model = DQN(
        env=env,
        policy_class=CnnPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        double_q=False,
        prioritized_replay=True,
        dueling=False,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        model_path='cartpole_model'
    )
    model.learn(total_timesteps=args.max_timesteps)
    model.save("cartpole_cnn_model")
    # model.load("cartpole_cnn_model")
    # model.evaluate(num_epsiodes=5)
    print("\nFinished")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=50000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
