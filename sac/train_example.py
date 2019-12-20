import argparse
import setGPU
import gym
import numpy as np

from sac import SAC

def main(args):
    """
    Train and save the SAC model, for the halfcheetah problem

    :param args: (ArgumentParser) the input arguments
    """
    env = gym.make(args.env)
    test_env = gym.make(args.env)

    if args.ent_coef is None:
        args.ent_coef = 'auto'

    model = SAC(env=env, 
                test_env=test_env,
                seed=int(args.seed),   
                ent_coef=args.ent_coef,
                reward_scale=5.
                )
    ep_rewards = model.learn(total_timesteps=int(args.max_timesteps),
                             save_path=args.save_path)

    model.save(args.save_path + "/%s_model_seed%d_fin_auto.zip"%(args.env, int(args.seed)))
    np.save(args.save_path + "/%s_rews_seed%d_fin_auto.npy"%(args.env, int(args.seed)), np.array(ep_rewards))
    
    # print("Saving model to halfcheetah_model.zip")
    # model.learn(total_timesteps=100)
    # model.load("halfcheetah_model.zip")

    model.evaluate(10)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SAC")
    parser.add_argument('--max-timesteps', default=2000000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--seed', default=1, type=int, help="Random seed for training")
    parser.add_argument('--env', default="HalfCheetah-v2")
    parser.add_argument('--ent_coef', default='auto')
    parser.add_argument('--reward_scale', default=1.)
    parser.add_argument('--save_path', default="results_final")

    args = parser.parse_args()
    print("- Environment : %s" % (args.env) )
    print("- Seed : %d" % (args.seed) )
    print("- Ent_coef : %s" % str(args.ent_coef) )

    main(args)

