import argparse

import gym
import numpy as np

from sac import SAC

def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("HalfCheetah-v2")

    model = SAC(env=env)
    model.learn(total_timesteps=args.max_timesteps)

    print("Saving model to cartpole_model.zip")
    
    model.save("cartpole_model.zip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
