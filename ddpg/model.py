import numpy as np
import tensorflow as tf

from lib.common import Buffer, BaseModel, Transition
from lib.tools import flatten


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


class DDPG(BaseModel):
    def __init__(self, name, sess, state_space, action_space, len_episode, gamma=0.98, lr=1e-3):
        super(DDPG, self).__init__(name, state_space, action_space)

        self.sess = sess
        self.gamma = gamma
        self.lr = lr
        self.act_dim = flatten(action_space)
        self.episode = Episode(len_episode)

        self.state_ph = tf.placeholder(tf.float32, (None,) + state_space, name='state-ph')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), name='act-ph')
        self.cum_reward_ph = tf.placeholder(tf.float32, (None,), name='cumulated-r')

        with tf.variable_scope('policy-net'):
            self.policy_net = self._construct(intput_ph=self.state_ph, out_dim=self.act_dim)

        with tf.name_scope('optimization'):
            act_one_hot = tf.one_hot(self.act_ph, self.act_dim)
            self.loss = -tf.reduce_sum(tf.nn.log_softmax(self.policy_net) * act_one_hot * self.cum_reward_ph, axis=1)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _construct(self, **kwargs):
        x = kwargs['input_ph']

        h = tf.layers.dense(x, units=100, activation=tf.nn.relu)
        h = tf.layers.dense(h, units=100, activation=tf.nn.relu)

        out = tf.layers.dense(h, units=kwargs['out_dim'])

        return out

    def store_transition(self, *args):
        self.episode.push(*args)

    def train(self, **kwargs):
        data = self.episode.sample(None)
        cum_reward = data.return_cum_reward(self.gamma)

        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.state_ph: data.state,
            self.cum_reward_ph: cum_reward
        })

        self.episode.clear()

        return loss
