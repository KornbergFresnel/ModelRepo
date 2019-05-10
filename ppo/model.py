import numpy as np
import tensorflow as tf

from lib.common import BaseModel
from lib.tools import Buffer, flatten


class Episode(Buffer):
    def __init__(self, len_episode):
        super(Episode, self).__init__(len_episode)

    def clear(self):
        self._data = []
        self._flag = 0

    def sample(self, batch_size):
        return Transition(*zip(*self._data))

    def return_cum_reward(self, gamma):
        reward = [None for _ in range(len(self._data))]
        reward[-1] = self._data[-1].reward

        for i in range(len(self._data) - 2, 0, -1):
            reward[i] = self._data[i].reward + self.gamma * self._data[i].reward

        return reward


class PPO(BaseModel):
    def __init__(self, name, sess, state_space, action_space, len_episode, actor_lr=1e-4, critic_lr=1e-3, gamma=0.96, epsilon=0.1, update_steps=5):
        super().__init__(name, state_space, action_space)

        self.sess = sess
        self.len_episode = len_episode

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_steps = update_steps
        self.act_dim = flatten(action_space)

        self.state_ph = tf.placeholder(tf.float32, (None,) + self.state_space, name='state-ph')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), name='action-ph')
        self.adv_ph = tf.placeholder(tf.float32, (None,), name='advantage-ph')
        self.cum_r_ph = tf.placeholder(tf.float32, (None,), name='cum-r-ph')

        with tf.variable_scope('new_policy'):
            self.new_a_scope = tf.get_variable_scope().name
            self.new_logits = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)

        with tf.variable_scope('old_policy'):
            self.old_a_scope = tf.get_variable_scope().name
            self.old_logits = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)

        with tf.variable_scope('critic'):
            self.value = self._construct(input_ph=self.state_ph, out_dim=1)

        with tf.name_scope('optimization'):
            ratio = self._policy(self.new_logits) / self._policy(self.old_logits)
            self.a_loss = (tf.stop_gradient(self.value) - tf.reshape(self.cum_r_ph, (-1, 1)))
            self.c_loss = tf.reduce_mean(tf.squared_difference(self.value, self.cum_r_ph))

            self.a_train_op = tf.train.AdamOptimizer(self.actor_lr).minimize(self.a_loss)
            self.c_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.c_loss)

        with tf.name_scope('update'):
            new_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.new_a_scope)
            old_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.old_a_scope)

            self.async = [tf.assign(t_var, e_var) for e_var, t_var in zip(new_vars, old_vars)]

        self.episode = Episode(len_episode)

    def _construct(self, **kwargs):
        h = tf.layers.dense(kwargs['input_ph'], units=100, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=100, activation=tf.nn.relu)
        out = tf.layers.dense(h, units=kwargs['out_dim'])

        return out

    def init(self):
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs):
        pass

    def store_transition(self, *args):
        self.episode.push(*args)

    def train(self, **kwargs):
        data = self.episode.sample(None)
        cum_reward = data.return_cum_reward(self.gamma)

        mean_a_loss, mean_c_loss = 0., 0.

        for _ in range(self.update_steps):
            c_loss = self.sess.run([self.c_loss, self.c_train_op], feed_dict={
                self.state_ph: data.state
                self.cum_r_ph: cum_reward
            })

            a_loss, _ = self.sess.run([self.a_loss, self.a_train_op], feed_dict={
                self.state_ph: data.state,
                self.cum_reward_ph: cum_reward
            })

            mean_a_loss += a_loss
            mean_c_loss += c_loss

        self.episode.clear()

        return mean_a_loss / self.update_steps, mean_c_loss / self.update_steps

