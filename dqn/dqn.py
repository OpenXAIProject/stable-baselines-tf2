import tensorflow as tf
import numpy as np
from gym.spaces import MultiDiscrete, Box

from tqdm import tqdm
from functools import partial
from common import tf_util, LinearSchedule
from common.vec_env import VecEnv

from base.rl import ValueBasedRLAlgorithm
from base.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

import copy
import gym

# For Save/Load
import os
import pickle
import cloudpickle
import json
import zipfile
from common.save_util import params_to_bytes
from common.save_util import data_to_json


class DQN(ValueBasedRLAlgorithm):
    def __init__(self, policy_class, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, 
                 exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,    
                 prioritized_replay_alpha=0.6, prioritized_replay_beta0=0.4, prioritized_replay_beta_iters=None,
                 prioritized_replay_eps=1e-6, _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 dueling=False,
                 model_path='~/params/'):

        #Create an instance for save and load path
        self.model_path = model_path
        # Create an instance of DQNPolicy (obs_space, act_space, n_env, n_steps, n_batch, name)
        self.env = env        
        self.observation_space = self.env.observation_space        
        self.action_space = self.env.action_space
        self.policy = policy_class(self.observation_space, self.action_space, 1, 1, None, 'q', dueling=dueling)
        self.q_function = self.policy.qnet.call                # Q-Function : obs -> action-dim vector        

        # Create another instance of DQNPolicy 
        self.target_policy = policy_class(self.observation_space, self.action_space, 1, 1, None, 'target_q', dueling=dueling)
        self.target_q_function = self.target_policy.qnet.call  # Q-Function : obs -> action-dim vector

        self.double_q = double_q
        if self.double_q:
            self.double_policy = policy_class(self.observation_space, self.action_space, 1, 1, None, 'double_q', dueling=dueling)
            self.double_q_function = self.double_policy.qnet.call

        self.buffer_size = buffer_size
        self.replay_buffer = None     

        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_eps = prioritized_replay_eps
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_iters = prioritized_replay_beta_iters

        self.num_timesteps = 0
        self.learning_starts = learning_starts        
        self.train_freq = train_freq        
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq        
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.learning_rate = learning_rate
        self.gamma = gamma        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.proba_step = self.policy.proba_step           
        self.exploration = None        
        self.episode_reward = None
        self.n_actions = self.action_space.nvec if isinstance(self.action_space, MultiDiscrete) else self.action_space.n

        self.qfunc_layers        = self.policy.qnet.trainable_layers
        self.target_qfunc_layers = self.target_policy.qnet.trainable_layers
        self.params              = self.qfunc_layers.trainable_variables + self.target_qfunc_layers.trainable_variables

        self.update_target()

    def act(self, obs, eps=1., stochastic=True):
        batch_size = np.shape(obs)[0]
        max_actions = np.argmax(self.q_function(obs), axis=1)

        if stochastic:                                  
            random_actions = np.random.randint(low=0, high=self.n_actions, size=batch_size)
            chose_random = np.random.uniform(size=np.stack([batch_size]), low=0, high=1) 
            epsgreedy_actions = np.where(chose_random < eps, random_actions, max_actions)
            
            return epsgreedy_actions

        else:
            return max_actions
    
    @tf.function                    
    def train(self, obs_t, act_t, rew_t, obs_tp, done_mask, importance_weights):                

        if self.double_q:
            q_tp1_best_using_online_net = tf.argmax(self.double_q_function(obs_tp), axis=1)
            q_tp1_best = tf.reduce_sum(self.target_q_function(obs_tp) 
                                       * tf.one_hot(q_tp1_best_using_online_net, self.n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(self.target_q_function(obs_tp), axis=1)

        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best
        q_t_selected_target = tf.cast(rew_t, tf.float32) + tf.cast(self.gamma, tf.float32) * q_tp1_best_masked

        with tf.GradientTape() as tape:           
            q_t_selected = tf.reduce_sum(self.q_function(obs_t) * tf.one_hot(act_t, self.n_actions), axis=1)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = tf_util.huber_loss(td_error)
            weighted_error = tf.reduce_mean(errors)       

        grads = tape.gradient(weighted_error, self.qfunc_layers.trainable_variables)    
        self.optimizer.apply_gradients(zip(grads, self.qfunc_layers.trainable_variables))

        return td_error, weighted_error
        
    def update_target(self):
        for var, var_target in zip(self.qfunc_layers, self.target_qfunc_layers):
            w = var.get_weights()
            var_target.set_weights(w)       

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN",
              reset_num_timesteps=True):

        # Create the replay buffer   
        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = total_timesteps
            else:
                prioritized_replay_beta_iters = self.prioritized_replay_beta_iters
            
            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.prioritized_replay_beta0,
                                                final_p=1.0)

        else:            
            self.replay_buffer = ReplayBuffer(self.buffer_size)                

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                          initial_p=1.0,
                                          final_p=self.exploration_final_eps)

        episode_rewards = [0.0]
        episode_successes = []

        saved_mean_rewards = None
        model_saved = False

        obs = self.env.reset()
        error = 0
        
        self.episode_reward = np.zeros((1,))

        for _ in tqdm(range(total_timesteps)):            
            # Take action and update exploration to the newest value            
            eps = self.exploration.value(self.num_timesteps)
            env_action = self.act(np.array(obs)[None], eps=0.1, stochastic=True)[0]
            new_obs, rew, done, info = self.env.step(env_action)

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, env_action, rew, new_obs, np.float32(done))
            obs = copy.deepcopy(new_obs)
            episode_rewards[-1] += rew

            if done:
                maybe_is_success = info.get('is_success')
                if maybe_is_success is not None:
                    episode_successes.append(float(maybe_is_success))
                if not isinstance(self.env, VecEnv):
                    obs = self.env.reset()
                episode_rewards.append(0.0)                
            
            can_sample = self.replay_buffer.can_sample(self.batch_size)

            if can_sample and self.num_timesteps > self.learning_starts:
                if self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.

                    # Sample a batch from the replay buffer
                    if self.prioritized_replay:
                        (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = \
                            self.replay_buffer.sample(self.batch_size,
                                                      beta=self.beta_schedule.value(self.num_timesteps))                        

                    else:
                        obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)                    
                        weights = np.ones_like(rewards)
                        batch_idxes = None

                    # Minimize the error in Bellman's equation on the sampled batch
                    td_errors, error = self.train(obses_t, actions, rewards, obses_tp1, dones, weights)                                                

                if self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target()

                if self.prioritized_replay:
                    new_priorities = np.abs(td_errors) + self.prioritized_replay_eps
                    self.replay_buffer.update_priorities(batch_idxes, new_priorities)

            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf

            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

            num_episodes = len(episode_rewards)

            if done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                print("- steps : ", self.num_timesteps)
                print("- episodes : ", num_episodes)
                print("- mean 100 episode reward : %.4f" % mean_100ep_reward)
                print("- recent mean TD error : %.4f" % error)                
                print("- % time spent exploring : ", int(100 * self.exploration.value(self.num_timesteps)))

                # Save if mean_100ep_reward is lager than the past best result
                if saved_mean_rewards == None or saved_mean_rewards < mean_100ep_reward:
                    self.save(self.model_path)
                    model_saved = True
                    saved_mean_rewards = mean_100ep_reward
                if model_saved:
                    print("best case: ", saved_mean_rewards)

            self.num_timesteps += 1

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)        
        observation = observation.reshape((-1,) + self.observation_space.shape)        
        actions, _, _ = self.policy.step(observation, deterministic=deterministic)
        actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if actions is not None:  # comparing the action distribution, to given actions
            actions = np.array([actions])
            assert isinstance(self.action_space, gym.spaces.Discrete)
            actions = actions.reshape((-1,))
            assert observation.shape[0] == actions.shape[0], "Error: batch sizes differ for actions and observations."
            actions_proba = actions_proba[np.arange(actions.shape[0]), actions]
            # normalize action proba shape
            actions_proba = actions_proba.reshape((-1, 1))

        if logp:
            actions_proba = np.log(actions_proba)
        
        actions_proba = actions_proba[0]

        return actions_proba

    def get_parameters(self):
        parameters = []
        weights = []
        for layer in self.params:
            weights.append(layer.get_weights())

        weights = np.array(weights)
        weights = weights.reshape(np.shape(self.params.trainable_variables))

        for idx, variable in enumerate(self.params.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=True):
        # params
        # data = {
        #     "double_q": self.double_q,
        #     "learning_starts": self.learning_starts,
        #     "train_freq": self.train_freq,
        #     "batch_size": self.batch_size,
        #     "target_network_update_freq": self.target_network_update_freq,
        #     "exploration_final_eps": self.exploration_final_eps,
        #     "exploration_fraction": self.exploration_fraction,
        #     "learning_rate": self.learning_rate,
        #     "gamma": self.gamma,
        #     "observation_space": self.observation_space,
        #     "action_space": self.action_space,
        #     "policy": self.policy
        #     "parameters"
        # }

        data = self.get_parameters()

        # self._save_to_file(model_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, parameters, exact_match=False):
        assert len(parameters) == len(self.params.weights)
        weights = []
        for variable, parameter in zip(self.params.weights, parameters):
            name, value = parameter
            if exact_match:
                assert name == variable.name
            weights.append(value)
        # print(weights)
        for i in range(len(self.params)):
            self.params[i].set_weights(weights[i])
        # self.qfunc_layers.set_weights(weights)

    def load(self, load_path, cloudpickle=True):
        # Parameter cloudpickle does not work now

        if isinstance(load_path, str):
            _, ext = os.path.splitext(load_path)
            if ext == "":
                load_path += ".pkl"
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        self.load_parameters(data)

        # self.double_q = data["double_q"]
        # self.learning_starts = data["learning_starts"]
        # self.train_freq = data["train_freq"]
        # self.batch_size = data["batch_size"]
        # self.target_network_update_freq = data["target_network_update_freq"]
        # self.exploration_final_eps = data["exploration_final_eps"]
        # self.exploration_fraction = data["exploration_fraction"]
        # self.learning_rate = data["learning_rate"]
        # self.gamma = data["gamma"]
        # self.observation_space = data["observation_space"]
        # self.action_space = data["action_space"]
        # self.policy = data["policy"]