# This implementation is based on codes of Jongmin Lee & Byeong-jun Lee
import time
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
import pickle
from base.rl import ActorCriticRLAlgorithm

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


@tf.function
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


class SquashedGaussianActor(tf.keras.layers.Layer):

    def __init__(self, env):
        super(SquashedGaussianActor, self).__init__()
        # obs_shape, action_dim, 
        self.obs_shape = env.observation_space.shape
        self.action_dim = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]

        # Actor parameters
        self.l1 = tf.keras.layers.Dense(64, activation='relu', name='f0', input_shape=(None,) + self.obs_shape)
        self.l2 = tf.keras.layers.Dense(64, activation='relu', name='f1')
        self.l3_mu = tf.keras.layers.Dense(self.action_dim, name='f2_mu')
        self.l3_log_std = tf.keras.layers.Dense(self.action_dim, name='f2_log_std')

    @tf.function
    def call(self, inputs, **kwargs):
        # obs = inputs
        h = self.l1(inputs)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.exp(log_std)
         
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)
        dist.shape = mean.shape
        sampled_action = dist.sample()
        sampled_action_logp = dist.log_prob(sampled_action)
        squahsed_action, squahsed_action_logp = apply_squashing_func(sampled_action, sampled_action_logp)

        return squahsed_action, tf.reshape(squahsed_action_logp, (-1,1))

    def dist(self, inputs):
        h = self.l1(inputs)
        h = self.l2(h)
        mean = self.l3_mu(h)
        log_std = self.l3_log_std(h)
        std = tf.exp(log_std)
        dist = tfp.distributions.MultivariateNormalDiag(mean, std)

        return dist

    def step(self, obs, deterministic=False):
        if deterministic:
            dist = self.dist(obs)            
            mean_action = dist.mean().numpy()
            mean_action_logp = dist.log_prob(mean_action)
            squashed_action, _ = apply_squashing_func(mean_action, mean_action_logp)            

        else:
            squahsed_action, _ = self.call(obs)
            squashed_action = squahsed_action.numpy()

        return squashed_action * self.max_action

class VNetwork(tf.keras.layers.Layer):

    def __init__(self, obs_shape, output_dim=1):
        super(VNetwork, self).__init__()

        self.v_l0 = tf.keras.layers.Dense(64, activation='relu', name='v/f0', input_shape=(None,) + obs_shape)
        self.v_l1 = tf.keras.layers.Dense(64, activation='relu', name='v/f1')
        self.v_l2 = tf.keras.layers.Dense(output_dim, name='v/f2')

    @tf.function
    def call(self, inputs, **kwargs):
        # obs, = inputs
        h = self.v_l0(inputs)
        h = self.v_l1(h)
        v = self.v_l2(h)
        return v


class QNetwork(tf.keras.layers.Layer):

    def __init__(self, obs_shape, num_critics=2):
        super(QNetwork, self).__init__()
        self.num_critics = num_critics

        self.qs_l0, self.qs_l1, self.qs_l2 = [], [], []
        for i in range(self.num_critics):
            self.qs_l0.append(tf.keras.layers.Dense(64, activation='relu', name='q%d/f0' % i, input_shape=(None,) + obs_shape))
            self.qs_l1.append(tf.keras.layers.Dense(64, activation='relu', name='q%d/f1' % i))
            self.qs_l2.append(tf.keras.layers.Dense(1, name='q%d/f2' % i))
    
    @tf.function
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


class SAC(ActorCriticRLAlgorithm):

    def __init__(self, env, policy_class=SquashedGaussianActor, ent_coef='auto', seed=0):
        super(SAC, self).__init__(policy_class=policy_class, env=env)
        tf.compat.v1.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.env = env
        self.max_action = self.env.action_space.high[0]
        self.obs_shape = self.env.observation_space.shape
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.replay_buffer = ReplayBuffer(self.max_action, buffer_size=100000)

        self.num_critics = 2
        self.gamma = 0.99
        self.tau = 0.005
        self.learning_rate = 3e-4
        self.batch_size = 64
        self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        self.ent_coef = ent_coef

        # self.optimizer_variables = []
        self.info_labels = ['actor_loss', 'v_loss', 'q_loss', 'mean(v)', 'mean(qs)', 'ent_coef', 'entropy', 'logp_pi']

        # Entropy coefficient (auto or fixed)
        if isinstance(self.ent_coef, str) and self.ent_coef == 'auto':
            # Default initial value of ent_coef when learned
            init_value = 1.0
            self.log_ent_coef = tf.keras.backend.variable(init_value, dtype=tf.float32, name='log_ent_coef')
            self.ent_coefficient = tf.exp(self.log_ent_coef)
            self.entropy_variables = [self.log_ent_coef]

        else:
            self.log_ent_coef = tf.math.log(self.ent_coef)
            self.ent_coefficient = tf.constant(self.ent_coef)

        # Actor, Critic Networks
        self.actor = policy_class(self.env)
        self.v = VNetwork(self.obs_shape)
        self.q = QNetwork(self.obs_shape, num_critics=self.num_critics)
        self.v_target = VNetwork(self.obs_shape)

        self.actor_variables = self.actor.trainable_variables
        self.critic_variables = self.v.trainable_variables + self.q.trainable_variables
        
        self.actor_optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.learning_rate)        

        if isinstance(ent_coef, str) and ent_coef == 'auto':
            self.entropy_optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)            

        self.optimizer_variables = self.actor.trainable_variables + self.v.trainable_variables + \
            self.q.trainable_variables + self.v_target.trainable_variables                    

    def update_target(self, target_params, source_params):
        for target, source in zip(target_params, source_params):
            tf.keras.backend.set_value(target, (1 - self.tau) * target + self.tau * source)        

    @tf.function
    def initialize_variables(self):
        zero_like_state = tf.zeros((1,) + self.obs_shape)
        zero_like_action = tf.zeros((1,self.action_dim))
        self.actor(zero_like_state)
        self.v(zero_like_state)
        self.v_target(zero_like_state)
        self.q(inputs=(zero_like_state, zero_like_action))       
                        
    @tf.function
    def train(self, obs, action, reward, next_obs, done):
        # Casting from float64 to float32
        obs = tf.cast(obs, tf.float32)
        action = tf.cast(action, tf.float32)
        reward = tf.cast(reward, tf.float32)
        next_obs = tf.cast(next_obs, tf.float32)
        done = tf.cast(done, tf.float32)

        dist = self.actor.dist(obs)

        with tf.GradientTape() as tape_actor:
            # Actor training (pi)
            action_pi, logp_pi = self.actor.call(obs)               
            qs_pi = self.q.call(inputs=(obs, action_pi))
            actor_loss = tf.reduce_mean(tf.math.exp(self.log_ent_coef) * logp_pi - tf.reduce_mean(qs_pi, axis=0))
            tape_actor.watch(qs_pi)
            tape_actor.watch(action_pi)
            tape_actor.watch(logp_pi)

        actor_variables = self.actor.trainable_variables
        grads_actor = tape_actor.gradient(actor_loss, actor_variables)
        actor_op = self.actor_optimizer.apply_gradients(zip(grads_actor, actor_variables))
        
        with tf.control_dependencies([actor_op]):            
            v_target = self.v_target(next_obs)
            min_q_pi = tf.reduce_min(qs_pi, axis=0) # (batch, 1)
            v_backup = tf.stop_gradient(min_q_pi - tf.math.exp(self.log_ent_coef) * logp_pi) # (batch, 1)
            q_backup = tf.stop_gradient(reward + (1 - done) * self.gamma * v_target)  # (batch, 1)

            with tf.GradientTape() as tape_critic:
                # Critic training (V, Q)
                v = self.v.call(obs)                
                v_loss = 0.5 * tf.reduce_mean((v_backup - v) ** 2)  # MSE, scalar
                
                qs = self.q.call(inputs=(obs, action))                                
                q_losses = [0.5 * tf.reduce_mean((q_backup - qs[k]) ** 2) for k in range(self.num_critics)] # (2, batch)
                q_loss = tf.reduce_sum(q_losses, axis=0)   # scalar

                value_loss = v_loss + q_loss
                
            critic_variables = self.v.trainable_variables + self.q.trainable_variables
            grads_critic = tape_critic.gradient(value_loss, critic_variables)
            self.critic_optimizer.apply_gradients(zip(grads_critic, critic_variables))

            if isinstance(self.ent_coef, str) and self.ent_coef == 'auto':
                with tf.GradientTape() as tape_ent:
                    ent_coef_loss = -tf.reduce_mean(self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))

                entropy_variables = [self.log_ent_coef]
                grads_ent = tape_ent.gradient(ent_coef_loss, entropy_variables)
                self.entropy_optimizer.apply_gradients(zip(grads_ent, entropy_variables))
        
        return actor_loss, tf.reduce_mean(v_loss), tf.reduce_mean(q_loss), tf.reduce_mean(v), tf.reduce_mean(qs), \
               tf.math.exp(self.log_ent_coef), tf.reduce_mean(dist.entropy()), tf.reduce_mean(logp_pi)

    def learn(self, total_timesteps, log_interval=2, seed=0, callback=None, verbose=1):
        np.random.seed(seed)
        
        self.initialize_variables()
        for target, source in zip(self.v_target.trainable_variables, self.v.trainable_variables):
            tf.keras.backend.set_value(target, source.numpy())        

        start_time = time.time()
        episode_rewards = [0.0]        

        obs = self.env.reset()
        for step in tqdm(range(total_timesteps), desc='SAC', ncols=70):
            if callback is not None:
                if callback(locals(), globals()) is False:
                    break

            # Take an action
            action = np.reshape(self.predict(np.array([obs]), deterministic=False)[0], -1)
            next_obs, reward, done, _ = self.env.step(action)

            # Store transition in the replay buffer.
            self.replay_buffer.add(obs, action, reward, next_obs, float(done))
            obs = next_obs

            episode_rewards[-1] += reward
            if done:
                obs = self.env.reset()
                episode_rewards.append(0.0)

            if self.replay_buffer.can_sample(self.batch_size):
                obss, actions, rewards, next_obss, dones = self.replay_buffer.sample(self.batch_size)  # action is normalize

                step_info = self.train(obss, actions, rewards, next_obss, dones)
                
                if verbose >= 1 and done and len(episode_rewards) % log_interval == 0:
                    print('\n============================')
                    print('%15s: %10.6f' % ('10ep_rewmean', np.mean(episode_rewards[-10:])))
                    for i, label in enumerate(self.info_labels):
                        print('%15s: %10.6f' %(label, step_info[i].numpy()))
                    print('============================\n')

                self.update_target(self.v_target.trainable_variables, self.v.trainable_variables)

        return episode_rewards

    def predict(self, obs, deterministic=False):
        obs_rank = len(obs.shape)
        if len(obs.shape) == 1:
            obs = np.array([obs])
        assert len(obs.shape) == 2
        
        action = self.actor.step(obs, deterministic=deterministic)
        
        if obs_rank == 1:
            return action[0], None
        else:
            return action, None

