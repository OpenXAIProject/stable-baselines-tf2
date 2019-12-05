import tensorflow as tf
import numpy as np
from gym.spaces import Discrete, MultiDiscrete, Box

from tqdm import tqdm
from functools import partial
from stable_baselines_tf2.common import OffPolicyRLModel, tf_util, LinearSchedule
from stable_baselines_tf2.common.replay_buffer import ReplayBuffer
from stable_baselines_tf2.common.policies import BasePolicy, nature_cnn, register_policy
from stable_baselines_tf2.common.vec_env import VecEnv
import copy

class DQNPolicy(BasePolicy):
    """
    Policy object that implements a DQN policy

    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    """

    def __init__(self, ob_space, ac_space, n_env, n_steps, n_batch, name='q', reuse=False, scale=False, dueling=True):
        # DQN policies need an override for the obs placeholder, due to the architecture of the code
        super(DQNPolicy, self).__init__(ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale)
        assert isinstance(ac_space, Discrete), "Error: the action space for DQN must be of type gym.spaces.Discrete"
        self.n_actions = ac_space.n
        self.value_fn = None
        self.q_values = None
        self.dueling = dueling
        self.policy_proba = None

    def _setup_init(self):
        """
        Set up action probability
        """
        # assert self.q_values is not None
        # self.policy_proba = tf.nn.softmax(self.q_values)

    def step(self, obs, state=None, mask=None, deterministic=True):
        """
        Returns the q_values for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray int, np.ndarray float, np.ndarray float) actions, q_values, states
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step

        :param obs: (np.ndarray float or int) The current observation of the environment
        :param state: (np.ndarray float) The last states (used in recurrent policies)
        :param mask: (np.ndarray float) The last masks (used in recurrent policies)
        :return: (np.ndarray float) the action probability
        """
        raise NotImplementedError

class QNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, obs_shape, n_action, name='q', layer_norm=False, dueling=False, n_batch=None):        
        self.layer_norm = layer_norm        
        self.dueling = dueling

        self.l1 = tf.keras.layers.Dense(64, name=name+'/l1', activation='relu', input_shape=(n_batch,)+ obs_shape)
        self.l2 = tf.keras.layers.Dense(64, name=name+'/l2', activation='relu')
        self.lout = tf.keras.layers.Dense(n_action, name=name+'/out')
        self.trainable_layers = [self.l1, self.l2, self.lout]

    @tf.function
    def call(self, input):
        h = self.l1(input)
        h = self.l2(h)        
        q_out = self.lout(h)

        # TODO : Implement Dueling Network Here
        return q_out


class FeedForwardPolicy(DQNPolicy):
    def __init__(self, ob_space, ac_space, n_env, n_steps, n_batch, name='q', reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 layer_norm=False, dueling=False, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)
        if layers is None:
            layers = [64, 64]

        self.feature_extraction = feature_extraction
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self.kwargs = kwargs
        self.layer_norm = layer_norm
        self.activation_function = act_fun
        self.qnet = QNetwork(layers, self.ob_space.shape, self.n_actions, name, layer_norm, dueling, n_batch)

    # @tf.function
    def q_value(self, obs):
        self.q_values = self.qnet(obs)
        self.policy_proba = tf.nn.softmax(self.q_values, axis=-1)
        return self.qnet(obs)

    def step(self, obs, state=None, mask=None, deterministic=True):
        # q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        q_values = self.q_value(obs)
        actions_proba = self.policy_proba(obs)
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            # Unefficient sampling
            # TODO: replace the loop
            # maybe with Gumbel-max trick ? (http://amid.fish/humble-gumbel)
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):        
        return self.policy_proba(obs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a MLP (2 layers of 64)

    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self, ob_space, ac_space, n_env, n_steps, n_batch, name='q', 
                 reuse=False, dueling=True, **_kwargs):
        super(MlpPolicy, self).__init__(ob_space, ac_space, n_env, n_steps, n_batch, name, reuse,
                                        feature_extraction="mlp", dueling=dueling,
                                        layer_norm=False, **_kwargs)

class DQN:

    def __init__(self, policy_class, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, 
                 exploration_fraction=0.1, exploration_final_eps=0.02, train_freq=1, batch_size=32, double_q=True,
                 learning_starts=1000, target_network_update_freq=500, prioritized_replay=False,                 
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False):
        
        # Create an instance of DQNPolicy (obs_space, act_space, n_env, n_steps, n_batch, name)
        self.env = env        
        self.observation_space = self.env.observation_space        
        self.action_space = self.env.action_space
        self.policy = policy_class(self.observation_space, self.action_space, 1, 1, None, 'q') 
        # Q-Function : obs -> action-dim vector        
        self.q_value = self.policy.qnet.call

        # Create another instance of DQNPolicy 
        self.target_policy = policy_class(self.observation_space, self.action_space, 1, 1, None, 'target_q')
        # Q-Function : obs -> action-dim vector
        self.target_q_value = self.target_policy.qnet.call

        self.learning_starts = learning_starts        
        self.train_freq = train_freq        
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq        
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.num_timesteps = 0
        
        self.learning_rate = learning_rate
        self.gamma = gamma        
        self.double_q = double_q                
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.proba_step = None
        self.buffer_size = buffer_size
        self.replay_buffer = None        
        self.exploration = None
        self.params = None        
        self.episode_reward = None
        self.n_actions = self.action_space.nvec if isinstance(self.action_space, MultiDiscrete) else self.action_space.n

        self.qfunc_layers        = self.policy.qnet.trainable_layers
        self.target_qfunc_layers = self.target_policy.qnet.trainable_layers
        self.update_target()

    def setup_model(self):        
        assert not isinstance(self.action_space, Box), \
            "Error: DQN cannot output a gym.spaces.Box action space."

    def act(self, obs, eps=1., stochastic=True):
        batch_size = np.shape(obs)[0]
        max_actions = np.argmax(self.q_value(obs), axis=1)
        
        if stochastic:                                  
            random_actions = np.random.randint(low=0, high=self.n_actions, size=batch_size)
            chose_random = np.random.uniform(size=np.stack([batch_size]), low=0, high=1) 
            epsgreedy_actions = np.where(chose_random < eps, random_actions, max_actions)
            
            return epsgreedy_actions

        else:
            return max_actions
    
    # @tf.function                    
    def train(self, obs_t, act_t, rew_t, obs_tp, done_mask, importance_weights):                

        q_tp1_best = tf.reduce_max(self.target_q_value(obs_tp), axis=1)                                          # (batch_size,)
        q_tp1_best_masked = (1.0 - done_mask) * q_tp1_best                                                      # (batch_size,)
        q_t_selected_target = tf.cast(rew_t, tf.float32) + tf.cast(self.gamma, tf.float32) * q_tp1_best_masked  # (batch_size,)

        with tf.GradientTape() as tape:           
            q_t_selected = tf.reduce_sum(self.q_value(obs_t) * tf.one_hot(act_t, self.n_actions), axis=1)
            td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            errors = tf_util.huber_loss(td_error)
            weighted_error = tf.reduce_mean(errors)

        # print("1", tape.watched_variables())      
        # print("2", self.qfunc_layers.trainable_variables)

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
        self.replay_buffer = ReplayBuffer(self.buffer_size)        
        
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                          initial_p=1.0,
                                          final_p=self.exploration_final_eps)

        episode_rewards = [0.0]
        episode_successes = []

        obs = self.env.reset()
        error = 0
        # reset = True        
        self.episode_reward = np.zeros((1,))

        for _ in tqdm(range(total_timesteps)):            
            # Take action and update exploration to the newest value            
            eps         = self.exploration.value(self.num_timesteps)                                    
            # reset       = False
            env_action  = self.act(np.array(obs)[None], eps=0.1, stochastic=True)[0]                         

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
                # reset = True
            
            can_sample = self.replay_buffer.can_sample(self.batch_size)

            if can_sample and self.num_timesteps > self.learning_starts:
                if self.num_timesteps % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.                                    
                    obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)                    
                    weights = np.ones_like(rewards)

                    td_errors, error = self.train(obses_t, actions, rewards, obses_tp1, dones, weights)                                                

                if self.num_timesteps % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.update_target()

            if len(episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
            else:
                mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)              

            num_episodes = len(episode_rewards)
            if done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                print("- steps", self.num_timesteps)
                print("- episodes", num_episodes)
                print("- mean 100 episode reward", mean_100ep_reward)
                print("- recent error", error)                
                print("- % time spent exploring", int(100 * self.exploration.value(self.num_timesteps)))           

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

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def get_parameter_list(self):
        return self.params

    def save(self, save_path, cloudpickle=False):
        # params
        data = {
            "double_q": self.double_q,            
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,            
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,            
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,            
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,            
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters()

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
