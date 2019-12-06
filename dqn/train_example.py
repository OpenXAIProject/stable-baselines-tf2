import argparse

import gym
import numpy as np

from dqn import DQN, MlpPolicy

def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("CartPole-v0")

    model = DQN(
        env=env,
        policy_class=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        double_q=False,
        prioritized_replay=False,
        exploration_fraction=0.5,
        exploration_final_eps=0.02,
    )
    model.learn(total_timesteps=args.max_timesteps)

    print("Saving model to cartpole_model.zip")
    
    model.save("cartpole_model.zip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
