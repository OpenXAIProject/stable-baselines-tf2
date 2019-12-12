# Stable Baselines with TF 2.0

This repository is based on the original implementations of Stable Baselines. (https://github.com/hill-a/stable-baselines)

In this version, we pursuit following properties:
1. Easy to debug using Eager-execution and Tensorflow 2.0.
2. Easy to read, as simple as possible.

## Quick start
We recommend to use Anaconda virtual environment. 
With using `environment.yml`, you can easily install what you need to run!
(However, to run experiments in Mujoco simulation tasks, make sure that you have the license for it.)

To start, enter the following commands in a terminal:
```
git clone https://github.com/tzs930/stable_baselines_tf2.git
cd stable_baselines_tf2
conda env create -f environment.yml
```

## Progress
- Abstract classes : 
    - `base/rl.py` : Includes abstract classes for RL algorihtm
    - `base/policy.py` : Includes abstract classes for RL policy
    - `base/replay_buffer.py` :  Includes replay buffer implementation (which is for off-policy RL algorithms)
    - TBU : Tensorboard Writer 

- Algorithms : (Not working yet / Working but not verified / Verified (reproduced) )
    - DQN : Working but not verified
    - SAC : Working but not verified
    - DDPG : Not working yet
    - Algorithms planning to add : TRPO/PPO, GAIL
