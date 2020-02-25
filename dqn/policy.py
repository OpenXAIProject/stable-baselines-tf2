import tensorflow as tf
import numpy as np

from gym.spaces import Discrete
from base.policy import BasePolicy
from common.tf_util import conv, conv_to_fc, linear


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


class CNNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(CNNetwork, self).__init__()
        layer_conv1 = tf.keras.layers.Conv2D(name='c1', filters=32, kernel_size=8, strides=4, padding='valid',
                                             activation='relu',
                                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))

        layer_conv2 = tf.keras.layers.Conv2D(name='c2', filters=64, kernel_size=4, strides=2, padding='valid',
                                             activation='relu',
                                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))

        layer_conv3 = tf.keras.layers.Conv2D(name='c3', filters=64, kernel_size=3, strides=1, padding='valid',
                                             activation='relu',
                                             kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)))

        layer_flat = tf.keras.layers.Flatten(name='fc')

        layer_dense = tf.keras.layers.Dense(512, name='fc1', activation='relu')

        # layer_dropout = tf.keras.layers.Dropout(0.5)

        # self.model = [layer_conv1, layer_conv2, layer_conv3, layer_flat, layer_dense, layer_dropout]
        self.model = [layer_conv1, layer_conv2, layer_conv3, layer_flat, layer_dense]

    @tf.function
    def call(self, input):
        h = tf.cast(input, tf.float32)
        for layer in self.model:
            # print(layer.name)
            h = layer(h)
        return h


class QNetwork(tf.keras.layers.Layer):
    def __init__(self, layers, obs_shape, n_action, name='q', layer_norm=False, dueling=False, n_batch=None, activation='relu',
                 cnn_extractor=CNNetwork, feature_extraction="cnn"):
        super(QNetwork, self).__init__()
        self.layer_norm = layer_norm
        self.dueling = dueling
        self.layers = []
        self.layer_norms = []
        self.activation = activation

        self.feature_extraction = feature_extraction

        if self.feature_extraction != "cnn":
            for i, layersize in enumerate(layers):
                if i == 0:
                    layer = tf.keras.layers.Dense(layersize, name=name+'/l1',
                                                  activation=activation, input_shape=(n_batch,) + obs_shape)

                else:
                    layer = tf.keras.layers.Dense(layersize, name=name+'/l%d' % (i+1),
                                                  activation=activation)
                self.layers.append(layer)

                if self.layer_norm:
                    self.layer_norms_QNet.append(tf.keras.layers.LayerNormalization(epsilon=1e-4))

            self.layer_out = tf.keras.layers.Dense(n_action, name=name + '/out')
            self.trainable_layers = self.layers + [self.layer_out] + self.layer_norms

        else:
            self.cnn_extractor = cnn_extractor

            self.conv_net = CNNetwork()
            self.conv_layers = self.conv_net.model

            self.layer_out = tf.keras.layers.Dense(n_action, name=name + '/out')

            self.trainable_layers = self.conv_layers[0:3] + [self.conv_layers[4], self.layer_out]

        if self.dueling:
            self.layer_norms_VNet = []
            self.layers_VNet = []

            for i, layersize in enumerate(layers):
                if i == 0:
                    layer = tf.keras.layers.Dense(layersize, name=name+'/v/l1', activation=activation,
                                                         input_shape=(n_batch,) + obs_shape)
                else:
                    layer = tf.keras.layers.Dense(layersize, name=name + '/v/l%d' % (i+1), activation=activation)

                self.layers_VNet.append(layer)

                if self.layer_norm:
                    self.layer_norms_VNet.append(tf.keras.layers.LayerNormalization(epsilon=1e-4))

            self.layer_out_VNet = tf.keras.layers.Dense(1, name=name+'/v/out')
            self.trainable_layers = self.trainable_layers \
                                    + self.layers_VNet + [self.layer_out_VNet] + self.layer_norms_VNet


    @tf.function
    def call(self, input):
        if self.feature_extraction == "cnn":
            extracted_features = self.conv_net(input)
            h = extracted_features
        else:
            h = input
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if self.layer_norm:
                    h = self.layer_norms[i](h)
        action_scores = self.layer_out(h)

        # TODO : Implement Dueling Network Here
        if self.dueling:
            # Value Network
            if self.feature_extraction == "cnn":
                h = extracted_features
            else:
                h = input
            for i, layer in enumerate(self.layers_VNet):
                h = layer(h)
                if self.layer_norm:
                    h = self.layer_norms_VNet[i](h)

            state_scores = self.layer_out_VNet(h)

            action_scores_mean = tf.reduce_mean(action_scores, axis=1)
            action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, axis=1)

            q_out = state_scores + action_scores_centered
        else:
            q_out = action_scores

        return q_out


class FeedForwardPolicy(DQNPolicy):
    def __init__(self, ob_space, ac_space, n_env, n_steps, n_batch, name='q', reuse=False, layers=None,
                 cnn_extractor=CNNetwork, feature_extraction="mlp",
                 layer_norm=False, dueling=False, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse)
        if layers is None:
            layers = [64, 64]

        self.reuse = reuse
        self.kwargs = kwargs
        self.layer_norm = layer_norm
        self.activation_function = act_fun
        self.qnet = QNetwork(layers, self.ob_space.shape, self.n_actions, name, layer_norm, dueling, n_batch,
                             self.activation_function, cnn_extractor, feature_extraction)

    @tf.function
    def q_value(self, obs):
        self.q_values = self.qnet(obs)
        self.policy_proba = tf.nn.softmax(self.q_values, axis=-1)
        return self.qnet(obs)

    def step(self, obs, state=None, mask=None, deterministic=True):
        # q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        q_values = self.q_value(obs)
        # actions_proba = self.policy_proba(obs)
        actions_proba = tf.nn.softmax(q_values)
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


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements DQN policy, using a CNN (the nature CNN)
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, ob_space, ac_space, n_env, n_steps, n_batch, name='q',
                 reuse=False, dueling=True, **_kwargs):
        super(CnnPolicy, self).__init__(ob_space, ac_space, n_env, n_steps, n_batch, name, reuse,
                                        feature_extraction="cnn", dueling=dueling,
                                        layer_norm=False, **_kwargs)


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

