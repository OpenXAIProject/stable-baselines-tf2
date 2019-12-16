import argparse

import gym
import numpy as np

from sac import SAC

def main(args):
    """
    Train and save the SAC model, for the halfcheetah problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make("HalfCheetah-v2")

    model = SAC(env=env, seed=args.seed)    
    ep_rewards = model.learn(total_timesteps=args.max_timesteps)

    model.save("results/halfcheetah_model_seed%d.zip"%(args.seed))
    np.save('results/halfcheetah_rews_seed%d.npy'%(args.seed), np.array(ep_rewards))
    # np.save('results/halfcheetah_rews_seed%d.npy', np.array(eval_rewards))
    # print("Saving model to halfcheetah_model.zip")

    # model.learn(total_timesteps=100)
    # model.load("halfcheetah_model.zip")

    model.evaluate(50)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SAC on HalfCheetah")
    parser.add_argument('--max-timesteps', default=3000000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--seed', default=1, type=int, help="Random seed for training")
    args = parser.parse_args()
    main(args)
