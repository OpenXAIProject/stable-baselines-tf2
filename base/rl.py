from abc import ABC, abstractmethod
import os
import glob
import warnings
from collections import OrderedDict
import json
import zipfile

import pickle
import numpy as np
import gym
import tensorflow as tf

from common import set_global_seeds
from common.save_util import (
    is_json_serializable, data_to_json, json_to_data, params_to_bytes, bytes_to_params
)
from base.policy import ActorCriticPolicy
from common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv


class BaseRLAlgorithm(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """
    def __init__(self, policy_class, env):        
        self.policy_class = policy_class
        self.env = env                
        self.observation_space = None
        self.action_space = None                
        self.num_timesteps = 0        
        self.params = None        

        if env is not None:            
            self.observation_space = env.observation_space
            self.action_space = env.action_space    

    @abstractmethod
    def get_parameter_list(self):
        """
        Get tensorflow Variables of model's parameters

        This includes all variables necessary for continuing training (saving / loading).

        :return: (list) List of tensorflow Variables
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Get current model parameters as dictionary of variable name -> ndarray.

        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.get_parameter_list()
        parameter_values = self.params
        
        return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        return return_dictionary
    
    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) -> boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        raise NotImplementedError

    def evaluate(self, num_epsiodes=5):
        """
        Test the learned model in the given environment

        :param observation: (np.ndarray) the input observation
        :return: (np.ndarray) episode returns of num_epsiodes
        """
        
        episode_returns = []
        print("* Evaluating...")
        for i in range(num_epsiodes):
            done = False
            obs = self.env.reset()
            ret = 0
            while not done:
                action = self.predict(np.array([obs]), deterministic=True)[0]
                obs, rew, done, _ = self.env.step(action)
                ret += rew
            print("- Episode %3d : %6.3f" % (i+1, ret))
            episode_returns.append(ret)

        episode_returns = np.array(episode_returns)
        print("\n* Evaluation Result :")
        print("- Average of %d epsiode returns : %6.3f" % (num_epsiodes, np.mean(episode_returns)))
        print("- Stddev. of %d epsiode returns : %6.3f" % (num_epsiodes, np.std(episode_returns)))

        return episode_returns

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        """
        If ``actions`` is ``None``, then get the model's action probability distribution from a given observation.

        Depending on the action space the output is:
            - Discrete: probability for each possible action
            - Box: mean and standard deviation of the action output

        However if ``actions`` is not ``None``, this function will return the probability that the given actions are
        taken with the given parameters (observation, state, ...) on this model. For discrete action spaces, it
        returns the probability mass; for continuous action spaces, the probability density. This is since the
        probability mass will always be zero in continuous spaces, see http://blog.christianperone.com/2019/01/
        for a good explanation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param actions: (np.ndarray) (OPTIONAL) For calculating the likelihood that the given actions are chosen by
            the model for each of the given parameters. Must have the same number of actions and observations.
            (set to None to return the complete action probability distribution)
        :param logp: (bool) (OPTIONAL) When specified with actions, returns probability in log-space.
            This has no effect if actions is None.
        :return: (np.ndarray) the model's (log) action probability
        """
        raise NotImplementedError

    def load_parameters(self, parameters):
        """
        Load model parameters from a file or a dictionary

        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.

        This does not load agent's hyper-parameters.

        :param parameters: (list) A list containing parameter values
        """
        raise NotImplementedError       

    @abstractmethod
    def save(self, save_path):
        """
        Save the current parameters to file

        :param save_path: (str or file-like) The save location
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, load_path):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        """
        raise NotImplementedError()    
    

class ActorCriticRLAlgorithm(tf.keras.layers.Layer, BaseRLAlgorithm):
    """
    The base class for Actor critic model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param policy_base: (BasePolicy) the base policy used by this method (default=ActorCriticPolicy)
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    """

    def __init__(self, policy_class, env):
        super(ActorCriticRLAlgorithm, self).__init__(policy_class, env)
        
        self.initial_state = None
        self.step = None
        self.proba_step = None
        self.params = None    

        # Actor Network
        self.actor = None

        # Critic Network
        self.v = None
        self.q = None

    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass        

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    def get_parameters(self):
        parameters = []
        weights = self.get_weights()
        for idx, variable in enumerate(self.trainable_variables):
            weight = weights[idx]
            parameters.append((variable.name, weight))
        return parameters

    def load_parameters(self, parameters, exact_match=False):
        assert len(parameters) == len(self.weights)
        weights = []
        for variable, parameter in zip(self.weights, parameters):
            name, value = parameter
            if exact_match:
                assert name == variable.name
            weights.append(value)
        self.set_weights(weights)

    def save(self, filepath):
        parameters = self.get_parameters()
        with open(filepath, 'wb') as f:
            pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            parameters = pickle.load(f)
        self.load_parameters(parameters)


class ValueBasedRLAlgorithm(tf.keras.layers.Layer, BaseRLAlgorithm):
    """
    The base class for off policy RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param replay_buffer: (ReplayBuffer) the type of replay buffer
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    """

    def __init__(self, policy_class, env, replay_buffer=None):
        super(ValueBasedRLAlgorithm, self).__init__(policy_class, env)
    
    @abstractmethod
    def learn(self, total_timesteps, callback=None, seed=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    @abstractmethod
    def save(self, save_path):
        pass

    @abstractmethod
    def load(self, load_path):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Envrionment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        pass


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.tensorboard_log_path, "{}_{}".format(self.tb_log_name, latest_run_id))
            self.writer = tf.compat.v1.summary.FileWriter(save_path, graph=self.graph)
        return self.writer

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.tensorboard_log_path, self.tb_log_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if self.tb_log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.add_graph(self.graph)
            self.writer.flush()
