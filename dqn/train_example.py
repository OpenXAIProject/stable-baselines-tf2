import argparse

import gym

from dqn import DQN
from policy import MlpPolicy, CnnPolicy

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
        prioritized_replay=True,
        dueling=True,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        model_path='cartpole_model_test'
    )
    model.learn(total_timesteps=args.max_timesteps)
    model.evaluate(num_epsiodes=50)
    print("\nTrain Finished")
    model = DQN(
        env=env,
        policy_class=MlpPolicy,
        learning_rate=1e-3,
        buffer_size=50000,
        double_q=False,
        prioritized_replay=True,
        dueling=True,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        model_path='cartpole_model_test'
    )
    print("\nBefore Loading")
    model.evaluate(num_epsiodes=50)
    model.load("cartpole_model_test")
    model.evaluate(num_epsiodes=50)
    print("Finished")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on cartpole")
    parser.add_argument('--max-timesteps', default=30000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)
