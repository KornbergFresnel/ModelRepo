import numpy as np
import tensorflow as tf

from lib.common import BaseModel, Buffer
from lib.tools import flatten


class MASQ(BaseModel):
    def __init__(self, name, sess, state_space, action_space, lr=1e-4, gamma=0.99, tau=0.98, memory_size=10**6, batch_size=64):
        super().__init__(name, state_space, action_space)

        self.sess = sess
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.act_dim = flatten(action_space)

        self.state_ph = tf.placeholder(tf.float32, (None,) + state_space, name='state-ph')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), name='act-ph')

        with tf.variable_scope('eval'):
            self.eval_scope = tf.get_variable_scope().name
            self.eval_net = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)

        with tf.variable_scope('target'):
            self.target_scope = tf.get_variable_scope().name
            self.target_net = self._construct(input_ph=self.state_ph, out_dim=self.act_dim)

        with tf.name_scope('update'):
            eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.eval_scope)
            target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_scope)

            self.async_op = [tf.assign(t_var, e_var) for e_var, t_var in zip(eval_vars, target_vars)]
            self.soft_async_op = [tf.assign(t_var, self.tau * e_var + (1. - self.tau) * t_var) for e_var, t_var in zip(eval_vars, target_vars)]

        with tf.name_scope('optimization'):
            self.loss = None
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        self.replay_buffer = Buffer(self.memory_size)

    def _construct(self, **kwargs):
        raise NotImplementedError

    def init(self):
        self.sess.run([tf.global_variables_initializer(), self.async])

    def act(self, obs, factor=0):
        raise NotImplementedError

    def store_transition(self, *args):
        self.replay_buffer.push(*args)

    def train(self, **kwargs):
        self.sess.run(self.soft_async_op)
