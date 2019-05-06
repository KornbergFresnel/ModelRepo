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

        with tf.variable_scope('policy-net'):
            self.policy_net = self._construct(intput_ph=self.state_ph, out_dim=self.act_dim)

        with tf.name_scope('optimization'):
            # TODO(ming): implement loss function
            self.loss = None
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

        # TODO(ming): discounted reward
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.state_ph: data.state})

        self.episode.clear()

        return loss
