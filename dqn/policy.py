import tensorflow as tf
import numpy as np

from gym.spaces import Discrete
from base.policy import BasePolicy, nature_cnn, register_policy

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

        if self.layer_norm:
            self.l1_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-4)
            self.l2_layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-4)

        self.lout = tf.keras.layers.Dense(n_action, name=name+'/out')
        self.trainable_layers = [self.l1, self.l2, self.lout]

    @tf.function
    def call(self, input):
        h = self.l1(input)
        if self.layer_norm:
            h = self.l1_layer_norm(h)
        
        h = self.l2(h)     
        if self.layer_norm:   
            h = self.l2_layer_norm(h)

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

    @tf.function
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

