# This implementation is based on codes of Jongmin Lee & Byeong-jun Lee
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import pickle


class ReplayBuffer:

    def __init__(self, max_action, buffer_size):
        self.obs = []
        self.action = []
        self.reward = []
        self.next_obs = []
        self.done = []
        self.max_action = max_action
        self.buffer_size = buffer_size

    def add(self, obs, action, reward, next_obs, done):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)
        if len(self.obs) > self.buffer_size:
            self.obs.pop(0)
            self.action.pop(0)
            self.reward.pop(0)
            self.next_obs.pop(0)
            self.done.pop(0)

    def can_sample(self, batch_size):
        return len(self.obs) >= batch_size

    def sample(self, batch_size):
        """
        Return samples (action is normalized)
        """
        obs, action, reward, next_obs, done = [], [], [], [], []
        indices = np.random.randint(0, len(self.obs), size=batch_size)
        for idx in indices:
            obs.append(self.obs[idx])
            action.append(self.action[idx] / self.max_action)
            reward.append(self.reward[idx])
            next_obs.append(self.next_obs[idx])
            done.append(self.done[idx])
        return np.array(obs), np.array(action), np.array(reward)[:, None], np.array(next_obs), np.array(done)[:, None]


def apply_squashing_func(sample, logp):
    """
    Squash the ouput of the gaussian distribution and account for that in the log probability.
    :param sample: (tf.Tensor) Action sampled from Gaussian distribution
    :param logp: (tf.Tensor) Log probability before squashing
    """
    # Squash the output
    squashed_action = tf.tanh(sample)
    squashed_action_logp = logp - tf.reduce_sum(tf.math.log(1 - squashed_action ** 2 + 1e-6), axis=1)  # incurred by change of variable
    return squashed_action, squashed_action_logp


class Actor(tf.keras.layers.Layer):

    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.action_dim = action_dim

        # Actor parameters
        self.l1 = tf.keras.layers.Dense(64, activation='relu', name='f0')
        self.l2 = tf.keras.layers.Dense(64, activation='relu', name='f1')
        self.l3_mu = tf.keras.layers.Dense(action_dim, name='f2_mu')
        self.l3_log_std = tf.keras.layers.Dense(action_dim, name='f2_log_std')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.l1(obs)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape

        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)
        squahsed_action, squahsed_action_logp = apply_squashing_func(sampled_action, sampled_action_logp)

        return squahsed_action, squahsed_action_logp, dist


class VNetwork(tf.keras.layers.Layer):

    def __init__(self, output_dim=1):
        super(VNetwork, self).__init__()

        self.v_l0 = tf.keras.layers.Dense(64, activation='relu', name='v/f0')
        self.v_l1 = tf.keras.layers.Dense(64, activation='relu', name='v/f1')
        self.v_l2 = tf.keras.layers.Dense(output_dim, name='v/f2')

    def call(self, inputs, **kwargs):
        obs, = inputs
        h = self.v_l0(obs)
        h = self.v_l1(h)
        v = self.v_l2(h)
        return v


class QNetwork(tf.keras.layers.Layer):

    def __init__(self, num_critics=2):
        super(QNetwork, self).__init__()
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(64, activation='relu', name='q%d/f0' % i))
            self.qs_l1.append(tf.keras.layers.Dense(64, activation='relu', name='q%d/f1' % i))
            self.qs_l2.append(tf.keras.layers.Dense(1, name='q%d/f2' % i))

    def call(self, inputs, **kwargs):
        obs, action = inputs
        obs_action = tf.concat([obs, action], axis=1)
        qs = []
        for i in range(self.num_critics):
            h = self.qs_l0[i](obs_action)
            h = self.qs_l1[i](h)
            q = self.qs_l2[i](h)
            qs.append(q)

        return qs


class SAC(tf.keras.layers.Layer):

    def __init__(self, env, ent_coef='auto', seed=0):
        super(SAC, self).__init__()
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.max_action = self.env.action_space.high[0]
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.obs_ph = tf.keras.layers.Input(self.state_dim, name='obs')
        self.action_ph = tf.keras.layers.Input(self.action_dim, name='action')
        self.reward_ph = tf.keras.layers.Input(1, name='reward')
        self.terminal_ph = tf.keras.layers.Input(1, name='terminal')
        self.next_obs_ph = tf.keras.layers.Input(self.state_dim, name='next_obs')

        self.replay_buffer = ReplayBuffer(self.max_action, buffer_size=50000)

        self.num_critics = 2
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        self.ent_coef = ent_coef

        optimizer_variables = []

        # Entropy coefficient (auto or fixed)
        if isinstance(self.ent_coef, str) and self.ent_coef == 'auto':
            # Default initial value of ent_coef when learned
            init_value = 1.0
            self.log_ent_coef = tf.keras.backend.variable(init_value, dtype=tf.float32, name='log_ent_coef')
            self.ent_coef = tf.exp(self.log_ent_coef)
        else:
            self.ent_coef = tf.constant(self.ent_coef)

        # Actor, Critic
        self.actor = Actor(self.action_dim)
        self.v = VNetwork()
        self.q = QNetwork(num_critics=self.num_critics)
        self.v_target = VNetwork()

        # Actor training
        action_pi, logp_pi, dist = self.actor([self.obs_ph])
        qs_pi = self.q([self.obs_ph, action_pi])
        actor_loss = tf.reduce_mean(self.ent_coef * logp_pi - tf.reduce_mean(qs_pi, axis=0))
        actor_optimizer = tf.keras.AdamOptimizer(self.learning_rate)
        actor_train_op = actor_optimizer.minimize(actor_loss, var_list=self.actor.trainable_variables)
        optimizer_variables += actor_optimizer.variables()

        with tf.control_dependencies([actor_train_op]):
            # Critic training (V, Q)
            v = self.v([self.obs_ph])
            min_q_pi = tf.reduce_min(qs_pi, axis=0)
            v_backup = tf.stop_gradient(min_q_pi - self.ent_coef * logp_pi)
            v_loss = tf.losses.mean_squared_error(v_backup, v)

            v_target = self.v_target([self.next_obs_ph])
            qs = self.q([self.obs_ph, self.action_ph])
            q_backup = tf.stop_gradient(self.reward_ph + (1 - self.terminal_ph) * self.gamma * v_target)  # batch x 1
            q_losses = [tf.losses.mean_squared_error(q_backup, qs[k]) for k in range(self.num_critics)]
            q_loss = tf.reduce_sum(q_losses)

            value_loss = v_loss + q_loss
            critic_optimizer = tf.keras.AdamOptimizer(self.learning_rate)
            critic_train_op = critic_optimizer.minimize(value_loss, var_list=self.v.trainable_variables + self.q.trainable_variables)
            optimizer_variables += critic_optimizer.variables()

            with tf.control_dependencies([critic_train_op]):
                # Entropy temperature
                if isinstance(ent_coef, str) and ent_coef == 'auto':
                    ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                    entropy_optimizer = tf.keras.AdamOptimizer(learning_rate=self.learning_rate)
                    entropy_train_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                    optimizer_variables += entropy_optimizer.variables()
                else:
                    entropy_train_op = tf.no_op('entropy_train_no_op')

                # Update target network
                source_params = self.v.trainable_variables
                target_params = self.v_target.trainable_variables
                # target_update_op = [
                #     tf.assign(target, (1 - self.tau) * target + self.tau * source)
                #     for target, source in zip(target_params, source_params)
                # ]

        # Copy weights to target networks
        # self.sess = tf.keras.backend.get_session()
        # self.sess.run(tf.variables_initializer(optimizer_variables))
        # self.v_target.set_weights(self.v.get_weights())

        # self.step_ops = [actor_train_op, critic_train_op, target_update_op, entropy_train_op] + \
        #                 [actor_loss, v_loss, q_loss, tf.reduce_mean(v), tf.reduce_mean(qs)] + \
        #                 [self.ent_coef, tf.reduce_mean(dist.entropy()), tf.reduce_mean(logp_pi)]
        # self.info_labels = ['actor_loss', 'v_loss', 'q_loss', 'mean(v)', 'mean(qs)', 'ent_coef', 'entropy', 'logp_pi']

        # For action selection
        self.sampled_action = action_pi
        self.deterministic_action = dist.mean()
    
    def update_target(self, target_params, source_params):
        for target, source in zip(target_params, source_params):
            target.set_weights( (1 - self.tau) * target.get_weights() + self.tau * source.get_weights() ) 
                    

    def train(self):
        obs, action, reward, next_obs, done = self.replay_buffer.sample(self.batch_size)  # action is normalized

        # step_result = self.sess.run(self.step_ops, feed_dict={
        #     self.obs_ph: obs,
        #     self.action_ph: action,
        #     self.reward_ph: reward,
        #     self.next_obs_ph: next_obs,
        #     self.terminal_ph: done
        # })

        return step_result[4:]

    def learn(self, total_timesteps, log_interval, seed, callback, verbose=1):
        np.random.seed(seed)

        start_time = time.time()
        episode_rewards = [0.0]

        obs = self.env.reset()
        for step in tqdm(range(total_timesteps), desc='SAC', ncols=70):
            if callback is not None:
                if callback(locals(), globals()) is False:
                    break

            # Take an action
            action = self.predict(np.array([obs]), deterministic=False)[0].flatten()
            next_obs, reward, done, info = self.env.step(action)

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs

            episode_rewards[-1] += reward
            if done:
                obs = self.env.reset()
                episode_rewards.append(0.0)

            if self.replay_buffer.can_sample(self.batch_size):
                step_info = self.train()
                if verbose >= 1 and done and len(episode_rewards) % log_interval == 0:
                    print('\n============================')
                    print('%12s: %10.3f' % ('ep_rewmean', np.mean(episode_rewards[-100:])))
                    for i, label in enumerate(self.info_labels):
                        print('%12s: %10.3f' %(label, step_info[i]))
                    print('============================\n')

        return episode_rewards

    def predict(self, obs, deterministic=False):
        obs_rank = len(obs.shape)
        if len(obs.shape) == 1:
            obs = np.array([obs])
        assert len(obs.shape) == 2

        if deterministic:
            action = self.sess.run(self.deterministic_action, feed_dict={self.obs_ph: obs})
        else:
            action = self.sess.run(self.sampled_action, feed_dict={self.obs_ph: obs})

        rescaled_action = action * self.max_action

        if obs_rank == 1:
            return rescaled_action[0], None
        else:
            return rescaled_action, None

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

    @staticmethod
    def load(filepath, env, seed):
        with open(filepath, 'rb') as f:
            parameters = pickle.load(f)

        model = SAC(env, seed=seed)
        model.load_parameters(parameters)
        return model